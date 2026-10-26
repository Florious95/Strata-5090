// Hardware bit contract: independent image banks vs explicit per-slot text deltas, and (pattern 3,
// C4 Row Table/Solo) identity banks vs a null table with cell positions, prompt cells included.
// Uses production native RoPE/norm+RoPE APIs; no model/encoder/GGUF.
#include "strata/kernels/mrope.hpp"
#include "strata/kernels/native_rope.hpp"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
namespace k = strata::kernels;
void ck(cudaError_t e) { if(e!=cudaSuccess){std::fprintf(stderr,"CUDA: %s\n",cudaGetErrorString(e));std::exit(2);} }
template<class T> struct Buffer {
    T* p=nullptr; size_t n;
    explicit Buffer(size_t size):n(size){ck(cudaMalloc(&p,n*sizeof(T)));}
    ~Buffer(){cudaFree(p);}
    void upload(const std::vector<T>& v){ck(cudaMemcpy(p,v.data(),v.size()*sizeof(T),cudaMemcpyHostToDevice));}
    std::vector<T> read(){std::vector<T> v(n);ck(cudaMemcpy(v.data(),p,n*sizeof(T),cudaMemcpyDeviceToHost));return v;}
};
int main(){
    int devices=0; if(cudaGetDeviceCount(&devices)!=cudaSuccess||!devices)return 77;
    constexpr int cells=262208, max_rows=12;
    Buffer<int32_t> banks[2] = {Buffer<int32_t>(size_t(cells)*3),Buffer<int32_t>(size_t(cells)*3)};
    Buffer<int32_t> cell_pos(max_rows),text_pos(max_rows);
    Buffer<float> input(size_t(max_rows)*512),gamma(256),reference(size_t(max_rows)*256),candidate(size_t(max_rows)*256);
    cudaStream_t streams[2];for(auto& s:streams)ck(cudaStreamCreateWithFlags(&s,cudaStreamNonBlocking));
    std::vector<float> x(input.n),g(gamma.n);
    for(size_t i=0;i<x.size();++i)x[i]=float(int(i%97)-48)/53.0f;
    for(size_t i=0;i<g.size();++i)g[i]=0.5f+float(i%31)/47.0f;
    input.upload(x);gamma.upload(g);
    int fixtures=0,replays=0;
    for(int pattern=0;pattern<4;++pattern){
        const bool text=pattern==3;
        const int delta[2]={pattern==0?-3:0,pattern>=2?0:-8};
        for(int slot=0;slot<2;++slot){
            std::vector<int32_t> tab(banks[slot].n);
            for(int c=0;c<cells;++c)for(int axis=0;axis<3;++axis)tab[size_t(c)*3+axis]=c>=20?c+delta[slot]:c;
            // The prefill bank really has unequal image t/h/w, not a fake scalar table.
            if(!text)for(int c=4;c<16;++c){tab[c*3]=4;tab[c*3+1]=4+(c-4)/3;tab[c*3+2]=4+(c-4)%3;}
            banks[slot].upload(tab);
        }
        for(int scale=0;scale<3;++scale){
            k::RopeScaling sc;
            if(scale){sc.type=scale==1?k::RopeScalingType::Linear:k::RopeScalingType::YaRN;sc.factor=2;sc.ext_factor=scale==2?1:0;}
            for(int base:{0,32,32768,258000})for(int n:{1,7,8,9,10,11,12})for(int hd:{128,256})for(int fused:{0,1})for(int narrow:{0,1}){
                if(base==0&&!text)continue;   // image cells: only the bank is right there
                const int split=(n+1)/2,stride=fused?2*hd:hd;
                std::vector<int32_t> cp(max_rows),tp(max_rows);
                for(int t=0;t<n;++t){int slot=t<split?0:1;cp[t]=base+t;tp[t]=cp[t]+delta[slot];}
                cell_pos.upload(cp);text_pos.upload(tp);
                auto apply=[&](int first,int count,bool ref,cudaStream_t stream){
                    auto* out=(ref?reference.p:candidate.p)+size_t(first)*hd;
                    const auto* pos=(ref?cell_pos.p:text_pos.p)+first;
                    const float* in=input.p+size_t(first)*stride;
                    if(fused)k::native_qsa_rms_norm_rope(in,stride,gamma.p,out,count,hd,64,1e-6f,sc,pos,stream);
                    else k::native_rope_apply(in,out,count,hd,64,sc,pos,stream);
                };
                auto record=[&](int first,int count,cudaStream_t stream){
                    for(int t=first;t<first+count;++t){
                        const int slot=t<split?0:1;
                        k::MropeTableScope own(banks[slot].p);apply(t,1,true,stream);
                    }
                    k::MropeTableScope explicit_text(nullptr);apply(first,count,false,stream);
                };
                auto compare=[&](){
                    for(auto s:streams)ck(cudaStreamSynchronize(s));
                    auto a=reference.read(),b=candidate.read();
                    if(std::memcmp(a.data(),b.data(),size_t(n)*hd*sizeof(float))){
                        std::fprintf(stderr,"BITS FAIL pattern=%d scale=%d base=%d rows=%d hd=%d fused=%d narrow=%d\n",pattern,scale,base,n,hd,fused,narrow);std::exit(1);
                    }
                };
                if(narrow){record(0,split,streams[0]);if(split<n)record(split,n-split,streams[1]);}
                else record(0,n,streams[0]);
                compare();
                cudaGraphExec_t exec[2]{};
                const int branches=narrow&&split<n?2:1;
                for(int b=0;b<branches;++b){
                    int first=b?split:0,count=narrow?(b?n-split:split):n;
                    ck(cudaStreamBeginCapture(streams[b],cudaStreamCaptureModeThreadLocal));
                    record(first,count,streams[b]);
                    cudaGraph_t graph;ck(cudaStreamEndCapture(streams[b],&graph));
                    ck(cudaGraphInstantiate(&exec[b],graph,0));ck(cudaGraphDestroy(graph));
                }
                for(int replay=0;replay<3;++replay){
                    // A different global pointer cannot change already captured slot arguments.
                    k::mrope_table_set(banks[replay%2].p);
                    for(int b=0;b<branches;++b)ck(cudaGraphLaunch(exec[b],streams[b]));
                    compare();++replays;
                }
                for(auto e:exec)if(e)ck(cudaGraphExecDestroy(e));
                ++fixtures;
            }
        }
    }
    k::mrope_table_set(nullptr);
    for(auto s:streams)ck(cudaStreamDestroy(s));
    std::printf("slot_mrope_gpu: fixtures=%d graph_replays=%d rows=1,7,8,9,10,11,12 text_banks=1 bits=0 PASS\n",fixtures,replays);
}
