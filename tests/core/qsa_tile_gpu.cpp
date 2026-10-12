// Synthetic full-Q4 two-slot contract: production GPU APIs, no model/assets.
#include "strata/kernels/qsa.hpp"
#include "strata/kernels/qsa_select.hpp"
#include "strata/kernels/qsa_decode_attn.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace k = strata::kernels;
void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) { std::fprintf(stderr,"%s: %s\n",what,cudaGetErrorString(e)); std::exit(2); }
}
template<class T> struct Buffer {
    T* p = nullptr; size_t size;
    explicit Buffer(size_t n) : size(n) { ck(cudaMalloc(&p,n*sizeof(T)),"alloc"); }
    ~Buffer() { cudaFree(p); }
    Buffer(const Buffer&) = delete;
    void upload(const std::vector<T>& v) { ck(cudaMemcpy(p,v.data(),v.size()*sizeof(T),cudaMemcpyHostToDevice),"upload"); }
    std::vector<T> download() const {
        std::vector<T> v(size); ck(cudaMemcpy(v.data(),p,size*sizeof(T),cudaMemcpyDeviceToHost),"download");return v;
    }
    void poison(cudaStream_t s) { ck(cudaMemsetAsync(p,0x7f,size*sizeof(T),s),"poison"); }
};
float random_f(uint32_t& s) { s^=s<<13; s^=s>>17; s^=s<<5; return float(int32_t(s))/2147483648.0f; }

struct Slot {
    k::QsaShapes shape = k::qsa_real_shapes();
    static constexpr int capacity = 262144, max_q = 8;
    int blocks = capacity/4+2;
    int cap = (int)k::qsa_selection_width(k::kTopkMaxCells,shape);
    size_t stride = k::qsa_decode_attn_scratch_floats(cap,shape);
    Buffer<float> pooled{size_t(blocks)*128}, dead{128}, qi{max_q*512}, q{size_t(max_q)*24*256};
    Buffer<int32_t> page{capacity/4}, steps{max_q*k::kStepCount};
    Buffer<uint8_t> kp{size_t(capacity)*2*256/32*18}, vp{size_t(capacity)*2*256/32*18};
    Buffer<float> a_score{size_t(max_q)*blocks}, b_score{size_t(max_q)*blocks};
    Buffer<int32_t> a_ids{size_t(max_q)*cap}, b_ids{size_t(max_q)*cap};
    Buffer<float> a_scratch{max_q*stride}, b_scratch{max_q*stride};
    Buffer<float> a_out{size_t(max_q)*24*256}, b_out{size_t(max_q)*24*256};
    k::QsaAttnPools pools{};
    int nq = 1;
    explicit Slot(uint32_t seed) {
        std::vector<float> v(pooled.size);for(auto& x:v)x=random_f(seed);pooled.upload(v);
        v.resize(dead.size);for(auto& x:v)x=random_f(seed);dead.upload(v);
        v.resize(qi.size);for(auto& x:v)x=random_f(seed);qi.upload(v);
        v.resize(q.size);for(auto& x:v)x=0.125f*random_f(seed);q.upload(v);
        std::vector<int32_t> pages(page.size);for(size_t i=0;i<pages.size();++i)pages[i]=(int32_t)i;page.upload(pages);
        std::vector<uint8_t> bytes(kp.size);
        for(size_t i=0;i<bytes.size();i+=18) {
            bytes[i]=0;bytes[i+1]=0x28;
            for(int j=2;j<18;++j) { seed^=seed<<13;seed^=seed>>17;seed^=seed<<5;bytes[i+j]=(uint8_t)seed; }
        }
        kp.upload(bytes);for(size_t i=0;i<bytes.size();i+=18)for(int j=2;j<18;++j)bytes[i+j]^=0x3c;vp.upload(bytes);
        pools.page_table=page.p;pools.k_q4=kp.p;pools.v_q4=vp.p;
    }
    void positions(int end) {
        std::vector<int32_t> s(steps.size);
        for(int t=0;t<max_q;++t) {
            int n=end-nq+t+1;int32_t* r=s.data()+t*k::kStepCount;
            r[k::kStepPos]=n-1;r[k::kStepNKv]=n;r[k::kStepNBid]=n/4;
            r[k::kStepWidth]=(int32_t)k::qsa_selection_width(n,shape);
        }
        steps.upload(s);
    }
    void poison(cudaStream_t stream) {
        a_score.poison(stream);b_score.poison(stream);a_ids.poison(stream);b_ids.poison(stream);
        a_scratch.poison(stream);b_scratch.poison(stream);a_out.poison(stream);b_out.poison(stream);
    }
    void run(cudaStream_t stream) {
        // Legacy batch_rec_ path: independent launches for each causal row.
        for(int t=0;t<nq;++t) {
            k::qsa_block_scores(pooled.p,dead.p,qi.p+t*512,steps.p+t*k::kStepCount,1,blocks,shape,a_score.p+size_t(t)*blocks,stream);
            k::qsa_block_topk(a_score.p+size_t(t)*blocks,steps.p+t*k::kStepCount,1,blocks,cap,shape,a_ids.p+size_t(t)*cap,stream);
            k::qsa_decode_attn_batch(q.p+size_t(t)*24*256,pools,a_ids.p+size_t(t)*cap,steps.p+t*k::kStepCount,cap,shape,a_scratch.p+t*stride,a_out.p+size_t(t)*24*256,1,stream);
        }
        // Candidate: same slot/snapshot, same per-query steps, existing MQ<=8 APIs.
        k::qsa_block_scores(pooled.p,dead.p,qi.p,steps.p,nq,blocks,shape,b_score.p,stream);
        k::qsa_block_topk(b_score.p,steps.p,nq,blocks,cap,shape,b_ids.p,stream);
        k::qsa_decode_attn_batch(q.p,pools,b_ids.p,steps.p,cap,shape,b_scratch.p,b_out.p,nq,stream);
    }
    template<class T> void compare(const Buffer<T>& a,const Buffer<T>& b,const char* name) {
        const auto x=a.download(),y=b.download();
        if(std::memcmp(x.data(),y.data(),x.size()*sizeof(T))) {
            size_t i=0;while(i<x.size()&&!std::memcmp(&x[i],&y[i],sizeof(T)))++i;
            std::fprintf(stderr,"bit mismatch stage=%s nq=%d element=%zu\n",name,nq,i);std::exit(1);
        }
    }
    void check() {
        compare(a_score,b_score,"scores+poison");compare(a_ids,b_ids,"ids+poison");compare(a_out,b_out,"attn+poison");
        const auto out=b_out.download();
        for(size_t i=0;i<size_t(nq)*24*256;++i)if(!std::isfinite(out[i])){std::fprintf(stderr,"nonfinite attn\n");std::exit(1);}
        const auto ids=b_ids.download();const auto s=steps.download();
        for(int t=0;t<nq;++t)for(int i=0;i<s[t*k::kStepCount+k::kStepWidth];++i)
            if(ids[size_t(t)*cap+i]<0||ids[size_t(t)*cap+i]>=s[t*k::kStepCount+k::kStepNKv]){std::fprintf(stderr,"causal selection violation\n");std::exit(1);}
    }
};

int main() {
    Slot left(0x174ab3),right(0x831ac9);
    cudaStream_t main{},side{};cudaEvent_t fork{},join{};
    ck(cudaStreamCreateWithFlags(&main,cudaStreamNonBlocking),"main");ck(cudaStreamCreateWithFlags(&side,cudaStreamNonBlocking),"side");
    ck(cudaEventCreateWithFlags(&fork,cudaEventDisableTiming),"fork");ck(cudaEventCreateWithFlags(&join,cudaEventDisableTiming),"join");
    const int widths[][2]={{1,1},{2,2},{3,3},{4,4},{5,5},{6,6},{7,5},{8,4},{1,8},{2,8},{3,8}};
    int fixtures=0,replays=0,width_calls[9]={};
    for(const auto& pair:widths)for(int end:{32,2051,32768,262141}) {
        left.nq=pair[0];right.nq=pair[1];left.positions(end);right.positions(end-3);
        left.poison(main);right.poison(main);ck(cudaStreamSynchronize(main),"poison direct");
        left.run(main);right.run(side);ck(cudaStreamSynchronize(main),"left direct");ck(cudaStreamSynchronize(side),"right direct");left.check();right.check();
        cudaGraph_t graph{};cudaGraphExec_t exec{};
        ck(cudaStreamBeginCapture(main,cudaStreamCaptureModeGlobal),"capture");
        ck(cudaEventRecord(fork,main),"fork record");ck(cudaStreamWaitEvent(side,fork,0),"fork wait");
        left.run(main);right.run(side);ck(cudaEventRecord(join,side),"join record");ck(cudaStreamWaitEvent(main,join,0),"join wait");
        ck(cudaStreamEndCapture(main,&graph),"end capture");ck(cudaGraphInstantiate(&exec,graph,0),"instantiate");
        for(int r=0;r<3;++r) {
            // Reuse graph/storage at changed causal positions, with every inactive row poisoned.
            left.positions(end+r);right.positions(end-3+r);left.poison(main);right.poison(main);
            ck(cudaGraphLaunch(exec,main),"replay");ck(cudaStreamSynchronize(main),"replay sync");left.check();right.check();++replays;
        }
        ck(cudaGraphExecDestroy(exec),"destroy exec");ck(cudaGraphDestroy(graph),"destroy graph");
        ++fixtures;++width_calls[left.nq];++width_calls[right.nq];
    }
    ck(cudaEventDestroy(fork),"destroy fork");ck(cudaEventDestroy(join),"destroy join");
    ck(cudaStreamDestroy(main),"destroy main");ck(cudaStreamDestroy(side),"destroy side");
    std::printf("PASS: %d two-slot Q4 fixtures, %d poisoned fork/join graph replays; bitwise scores/IDs/attn; causal IDs\n",fixtures,replays);
    for(int i=1;i<=8;++i)std::printf("width=%d direct=%d graph=%d\n",i,width_calls[i],3*width_calls[i]);
}
