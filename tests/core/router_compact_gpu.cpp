// Public asset-free NVIDIA parity/capture test. Requires real CUDA execution;
// host-only syntax checking is not GPU qualification. No model/PCIe benchmark.
#include "strata/kernels/native_router.hpp"
#include <cuda_runtime.h>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <random>
#include <stdexcept>
#include <vector>

namespace k = strata::kernels;
namespace {
constexpr int maximum_rows = 17, guard = 16;
int rows = 0, pattern = 0, replay = 0;
void require(bool ok,const char* what) {
    if (!ok) { std::fprintf(stderr,"FAIL %s rows=%d pattern=%d replay=%d\n",what,rows,pattern,replay); std::exit(1); }
}
void ck(cudaError_t e,const char* what) {
    if (e != cudaSuccess) { std::fprintf(stderr,"%s: %s\n",what,cudaGetErrorString(e)); std::exit(2); }
}
template<class T> struct Buffer {
    T* p = nullptr;
    const size_t size;
    explicit Buffer(size_t n):size(n) { ck(cudaMalloc(reinterpret_cast<void**>(&p),size*sizeof(T)),"alloc"); }
    ~Buffer() { cudaFree(p); }
    Buffer(const Buffer&) = delete;
    T* data() const { return p+guard; }
    void poison(cudaStream_t stream) { ck(cudaMemsetAsync(p,0xff,size*sizeof(T),stream),"poison"); }
    std::vector<T> download() const {
        std::vector<T> v(size);ck(cudaMemcpy(v.data(),p,size*sizeof(T),cudaMemcpyDeviceToHost),"download");return v;
    }
};
template<class T> void equal(const Buffer<T>& a,const Buffer<T>& b,const char* label) {
    const auto x=a.download(),y=b.download();
    require(!std::memcmp(x.data(),y.data(),x.size()*sizeof(T)),label);
}
struct Graph {
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t exec = nullptr;
    ~Graph() { if(exec)cudaGraphExecDestroy(exec);if(graph)cudaGraphDestroy(graph); }
};
void route(const float* logits,int32_t* ids,float* weights,cudaStream_t stream) {
    if(rows==1)k::native_router_top10(logits,ids,weights,stream);
    else k::native_router_top10_multi(logits,ids,weights,rows,stream);
}
}

int main() {
    int devices=0;
    if(cudaGetDeviceCount(&devices)!=cudaSuccess||devices==0) {
        std::fprintf(stderr,"SKIP: NVIDIA CUDA unavailable\n");return 77;
    }
    cudaDeviceProp props{};ck(cudaGetDeviceProperties(&props,0),"device props");
    std::printf("GPU %s sm_%d%d\n",props.name,props.major,props.minor);
    cudaStream_t stream=nullptr;ck(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking),"stream");
    Buffer<float> input(2*guard+maximum_rows*512), old_w(2*guard+maximum_rows*10),
                  new_w(2*guard+maximum_rows*10), dispatch_w(2*guard+maximum_rows*10);
    Buffer<int32_t> old_ids(2*guard+maximum_rows*10),new_ids(2*guard+maximum_rows*10),
                    dispatch_ids(2*guard+maximum_rows*10);
    Buffer<uint32_t> diagnostics(2*guard+maximum_rows*2);
    std::map<int,std::unique_ptr<Graph>> graphs;
    auto rejects=[&](auto call) {
        bool rejected=false;
        try { call(); } catch(const std::invalid_argument&) { rejected=true; }
        require(rejected,"invalid compact operator arguments must reject before launch");
    };
    rejects([&] { k::native_router_top10_compact(input.data(),old_ids.data(),old_w.data(),0,stream); });
    rejects([&] { k::native_router_top10_compact(input.data(),old_ids.data(),old_w.data(),1,nullptr); });
    rejects([&] { k::native_router_top10_compact(input.data(),old_ids.data(),reinterpret_cast<float*>(old_ids.data()),1,stream); });
    rejects([&] { k::native_router_top10_compact(input.data(),old_ids.data(),old_w.data(),1,stream,reinterpret_cast<uint32_t*>(old_ids.data())); });
    rejects([&] { k::native_router_top10_compact(reinterpret_cast<const float*>(uintptr_t(input.data())+1),old_ids.data(),old_w.data(),1,stream); });
    std::mt19937 random(5090);
    size_t fixtures=0,compact_count=0,fallback_count=0;
    for(int n:{1,2,3,4,6,8,12,16,17}) {
        rows=n;
        for(pattern=0;pattern<10;++pattern)for(replay=0;replay<4;++replay) {
            std::vector<float> host(input.size);std::memset(host.data(),0xff,host.size()*sizeof(float));
            for(int row=0;row<rows;++row)for(int expert=0;expert<512;++expert) {
                float x=0;
                const int rotated=(expert/32+row+replay)%16;
                if(pattern==1)x=rotated==0?8.0f:-20.0f;             // exactly32 survivors
                if(pattern==2)x=rotated<4?8.0f:-20.0f;              // exactly128 survivors
                if(pattern==3)x=expert<129?8.0f:-20.0f;             // dense boundary => fallback
                if(pattern==4)x=expert<32?8.0f+(expert%32)*0x1p-14f:-20.0f;
                if(pattern==5)x=expert==0?80.0f:-100.0f;            // zero/subnormal safety
                if(pattern==6)x=expert<32?0.0f:-90.0f;              // subnormal probability or FTZ-zero
                if(pattern==7)x=(int(random()%4097)-2048)*0x1p-8f;
                if(pattern==8)x=expert%32==0?(expert==0?0.0f:-17.0f):(expert<32?-20.0f:-60.0f);
                if(pattern==9)x=expert<16?8.0f:-20.0f;              // exactly16 disjoint-group survivors
                host[guard+row*512+expert]=x;
            }
            ck(cudaMemcpy(input.p,host.data(),host.size()*sizeof(float),cudaMemcpyHostToDevice),"input");
            auto poison=[&] {
                old_ids.poison(stream);new_ids.poison(stream);dispatch_ids.poison(stream);
                old_w.poison(stream);new_w.poison(stream);dispatch_w.poison(stream);diagnostics.poison(stream);
                ck(cudaStreamSynchronize(stream),"poison sync");
            };
            poison();k::native_router_set_compact_enabled(false);
            route(input.data(),old_ids.data(),old_w.data(),stream);
            ck(cudaStreamSynchronize(stream),"legacy sync");
            if(!graphs.count(rows)) {
                // Warm the new kernel before capture (including lazy loading).
                k::native_router_top10_compact(input.data(),new_ids.data(),new_w.data(),rows,stream,diagnostics.data());
                ck(cudaStreamSynchronize(stream),"compact warm");
                auto graph=std::make_unique<Graph>();k::native_router_set_compact_enabled(true);
                ck(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal),"capture");
                k::native_router_top10_compact(input.data(),new_ids.data(),new_w.data(),rows,stream,diagnostics.data());
                route(input.data(),dispatch_ids.data(),dispatch_w.data(),stream); // actual production dispatch
                ck(cudaStreamEndCapture(stream,&graph->graph),"end capture");
                ck(cudaGraphInstantiate(&graph->exec,graph->graph,nullptr,nullptr,0),"instantiate");
                graphs.emplace(rows,std::move(graph));
                k::native_router_set_compact_enabled(false); // captured choice must not depend on this later
            }
            // Warmup may have touched candidate outputs; poison them again, not the oracle.
            new_ids.poison(stream);dispatch_ids.poison(stream);new_w.poison(stream);dispatch_w.poison(stream);
            diagnostics.poison(stream);
            ck(cudaGraphLaunch(graphs.at(rows)->exec,stream),"replay");ck(cudaStreamSynchronize(stream),"replay sync");
            equal(old_ids,new_ids,"bitwise ranked IDs+all inactive/guard bytes");
            equal(old_w,new_w,"bitwise FP32 weights+all inactive/guard bytes");
            equal(old_ids,dispatch_ids,"production switch ID parity");equal(old_w,dispatch_w,"production switch weight parity");
            const auto actual_input=input.download();
            require(!std::memcmp(host.data(),actual_input.data(),host.size()*sizeof(float)),"input read-only");
            const auto d=diagnostics.download();
            const auto ids=new_ids.download();
            const auto weights=new_w.download();
            for(size_t i=0;i<d.size();++i) {
                if(i<size_t(guard)||i>=size_t(guard+rows*2))require(d[i]==0xffffffffu,"diagnostic inactive/guard poison");
            }
            for(int row=0;row<rows;++row) {
                const uint32_t count=d[guard+row*2],used=d[guard+row*2+1];
                require(count>=16&&count<=512&&used<=1,"valid diagnostic record");
                if(pattern==0||pattern==3||pattern==5)require(used==0,"dense rows must use full fallback");
                if(pattern==6)require((count==512&&used==0)||(count==32&&used==1),"subnormals fall back, FTZ-zero may safely compact");
                if(pattern==8)require(count==47&&used==1,"multiple winners in one original lane must compact");
                if(pattern==1)require(count==32&&used==1,"sparse32 must actually compact");
                if(pattern==2)require(count==128&&used==1,"capacity128 must actually compact");
                if(pattern==9)require(count==16&&used==1,"group-bound16 must actually compact");
                compact_count+=used;fallback_count+=!used;
                float sum=0;
                for(int rank=0;rank<10;++rank) {
                    const int id=ids[guard+row*10+rank];const float w=weights[guard+row*10+rank];
                    require(id>=0&&id<512&&std::isfinite(w)&&w>=0,"valid finite expert output");sum+=w;
                    for(int prior=0;prior<rank;++prior)require(id!=ids[guard+row*10+prior],"unique expert IDs");
                }
                require(std::fabs(sum-1.0f)<2e-5f,"normalized weights");
            }
            ++fixtures;
        }
    }
    k::native_router_set_compact_enabled(false);
    require(compact_count>0&&fallback_count>0&&graphs.size()==9,"both executed branches and fixed graph reuse");
    graphs.clear();ck(cudaStreamDestroy(stream),"stream destroy");
    std::printf("PASS compact router GPU: %zu poisoned graph fixtures, %zu compact rows/%zu full fallback rows; "
                "exact IDs+weight bits, actual production dispatch, 9 reused graphs and tail guards\n",
                fixtures,compact_count,fallback_count);
    std::printf("NOT COVERED: full-model sampling, PCIe bytes/latency, end-to-end speed or 300MiB lifecycle floor\n");
}
