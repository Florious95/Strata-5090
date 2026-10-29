// Hardware bit contract: independent image banks vs explicit per-slot text deltas, and (pattern 3,
// C4 Row Table/Solo) identity banks vs a null table with cell positions, prompt cells included.
// C8-4 Vision8: the same contract over 8 banks (owners 4..7 first) and windows of up to 16 rows.
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
    constexpr int cells=262208, max_rows=16, max_banks=8;
    // generate.cpp's table: one allocation, slot b's bank at mrope_bank_offset(b, cells)
    Buffer<int32_t> table(k::mrope_bank_offset(max_banks,cells));
    auto bank=[&](int b){return table.p+k::mrope_bank_offset(b,cells);};
    Buffer<int32_t> cell_pos(max_rows),text_pos(max_rows);
    Buffer<float> input(size_t(max_rows)*512),gamma(256),reference(size_t(max_rows)*256),candidate(size_t(max_rows)*256);
    cudaStream_t streams[2];for(auto& s:streams)ck(cudaStreamCreateWithFlags(&s,cudaStreamNonBlocking));
    std::vector<float> x(input.n),g(gamma.n);
    for(size_t i=0;i<x.size();++i)x[i]=float(int(i%97)-48)/53.0f;
    for(size_t i=0;i<g.size();++i)g[i]=0.5f+float(i%31)/47.0f;
    input.upload(x);gamma.upload(g);
    int fixtures=0,replays=0,c8_fixtures=0;
    for(int nb:{2,max_banks}){
        const bool c8=nb==max_banks;
        // C4/2-bank: the original fixture set; C8: every owner, windows of 1..16 rows
        const std::vector<int> ns=c8?std::vector<int>{1,8,9,16}:std::vector<int>{1,7,8,9,10,11,12};
        for(int pattern=0;pattern<4;++pattern){
            if(c8&&(pattern==1||pattern==2))continue;
            const bool text=pattern==3;
            std::vector<int> delta(nb);
            if(!c8){delta[0]=pattern==0?-3:0;delta[1]=pattern>=2?0:-8;}
            else for(int b=0;b<nb;++b)delta[b]=text?0:-1-(b%5);
            for(int slot=0;slot<nb;++slot){
                std::vector<int32_t> tab(size_t(cells)*3);
                for(int c=0;c<cells;++c)for(int axis=0;axis<3;++axis)tab[size_t(c)*3+axis]=c>=20?c+delta[slot]:c;
                // The prefill bank really has unequal image t/h/w, not a fake scalar table (C8: widths 2..4 by owner).
                const int w=c8?2+slot%3:3;
                if(!text)for(int c=4;c<16;++c){tab[c*3]=4;tab[c*3+1]=4+(c-4)/w;tab[c*3+2]=4+(c-4)%w;}
                ck(cudaMemcpy(bank(slot),tab.data(),tab.size()*sizeof(int32_t),cudaMemcpyHostToDevice));
            }
            for(int scale=0;scale<3;++scale){
                if(c8&&scale==1)continue;
                k::RopeScaling sc;
                if(scale){sc.type=scale==1?k::RopeScalingType::Linear:k::RopeScalingType::YaRN;sc.factor=2;sc.ext_factor=scale==2?1:0;}
                for(int base:{0,32,32768,258000})for(int n:ns)for(int hd:{128,256})for(int fused:{0,1})for(int narrow:{0,1}){
                    if(base==0&&!text)continue;   // image cells: only the bank is right there
                    if(c8&&base==32768)continue;
                    const int split=(n+1)/2,stride=fused?2*hd:hd;
                    // row t's owner: C4 two halves; C8 contiguous runs from owner 4 (16 rows: 4,4,5,5,..,3,3)
                    auto owner=[&](int t){return c8?(4+t*nb/n)%nb:(t<split?0:1);};
                    std::vector<int32_t> cp(max_rows),tp(max_rows);
                    for(int t=0;t<n;++t){cp[t]=base+t;tp[t]=cp[t]+delta[owner(t)];}
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
                            k::MropeTableScope own(bank(owner(t)));apply(t,1,true,stream);
                        }
                        k::MropeTableScope explicit_text(nullptr);apply(first,count,false,stream);
                    };
                    auto compare=[&](){
                        for(auto s:streams)ck(cudaStreamSynchronize(s));
                        auto a=reference.read(),b=candidate.read();
                        if(std::memcmp(a.data(),b.data(),size_t(n)*hd*sizeof(float))){
                            std::fprintf(stderr,"BITS FAIL banks=%d pattern=%d scale=%d base=%d rows=%d hd=%d fused=%d narrow=%d\n",nb,pattern,scale,base,n,hd,fused,narrow);std::exit(1);
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
                        k::mrope_table_set(bank((replay*3+1)%nb));
                        for(int b=0;b<branches;++b)ck(cudaGraphLaunch(exec[b],streams[b]));
                        compare();++replays;
                    }
                    for(auto e:exec)if(e)ck(cudaGraphExecDestroy(e));
                    ++fixtures;
                    if(c8)++c8_fixtures;
                }
            }
        }
    }
    k::mrope_table_set(nullptr);
    for(auto s:streams)ck(cudaStreamDestroy(s));
    std::printf("slot_mrope_gpu: fixtures=%d graph_replays=%d rows=1,7,8,9,10,11,12 text_banks=1 c8_fixtures=%d c8_banks=8 c8_rows=1,8,9,16 bits=0 PASS\n",
                fixtures,replays,c8_fixtures);
}
