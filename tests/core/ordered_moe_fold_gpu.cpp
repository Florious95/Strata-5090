// Asset-free, fixed-leaf CPU/GPU reduction witness. The H2D and D2D producers
// copy IDENTICAL predefined expert outputs, not Q8_K/Q8_1 expert arithmetic.
// No claim of full-model, GDN, sampler, or cross-quantizer determinism.
#include "strata/kernels/native_moe.hpp"
#include "strata/kernels/ordered_moe_fold.hpp"
#include "strata/kernels/s2_expert_grouped.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <cfenv>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <numeric>
#include <random>
#include <utility>
#include <vector>

namespace k = strata::kernels;
namespace om = strata::kernels::ordered_moe;
namespace {
char context[192] = {};
size_t cases = 0, graph_replays = 0, fused_calls = 0, scatter_calls = 0;
[[noreturn]] void fail(const char* label) {
    std::fprintf(stderr,"FAIL ordered MoE GPU: %s (%s)\n",label,context); std::exit(1);
}
void require(bool ok, const char* label) { if (!ok) fail(label); }
void ck(cudaError_t error, const char* label) {
    if (error!=cudaSuccess) {
        std::fprintf(stderr,"%s: %s (%s)\n",label,cudaGetErrorString(error),context); std::exit(2);
    }
}
float fbits(uint32_t bits) { return std::bit_cast<float>(bits); }
template<class T> struct Buffer {
    T* p = nullptr; size_t n;
    explicit Buffer(size_t count) : n(count) { ck(cudaMalloc(reinterpret_cast<void**>(&p),n*sizeof(T)),"device alloc"); }
    ~Buffer() { cudaFree(p); }
    Buffer(const Buffer&) = delete;
    void upload(const std::vector<T>& values) {
        require(values.size()==n,"upload size");
        ck(cudaMemcpy(p,values.data(),n*sizeof(T),cudaMemcpyHostToDevice),"upload");
    }
    std::vector<T> download() const {
        std::vector<T> values(n); ck(cudaMemcpy(values.data(),p,n*sizeof(T),cudaMemcpyDeviceToHost),"download");return values;
    }
    void poison() { ck(cudaMemset(p,0xff,n*sizeof(T)),"poison"); }
};
struct Graph {
    cudaGraph_t raw = nullptr; cudaGraphExec_t exec = nullptr;
    ~Graph() { if (exec) cudaGraphExecDestroy(exec); if (raw) cudaGraphDestroy(raw); }
};
enum Variant { DirectShared, DirectNoShared, AllHits, Scatter, Fused, Gated, VariantCount };
const char* names[] = {"direct","no-shared","zero+hits","scatter+combine","fused-or-scatter","gated-zero"};
struct Phase { int rows; std::array<int,12> owners; };
const Phase phases[] = {
    {1,{3}}, {12,{0,0,0,0,0,0,3,3,3,3,3,3}}, {1,{3}},
    {12,{3,3,3,1,1,1,0,0,0,2,2,2}}, {1,{3}},
    {4,{3,1,0,2}}, {8,{3,3,3,3,1,1,1,1}}, {1,{1}},
};
struct Fixture {
    static constexpr int max_rows = 12, guard = 32;
    int width, experts;
    Buffer<float> parts, hits, gpu_source, weights, shared, gate, output;
    Buffer<float> oracle_parts, oracle_shared, oracle_output;
    Buffer<int32_t> destinations, count{1};
    float* cpu_source = nullptr;
    cudaStream_t main = nullptr, producer = nullptr;
    cudaEvent_t fork = nullptr, join = nullptr;
    std::map<std::pair<int,int>,std::unique_ptr<Graph>> graphs;
    std::vector<float> expected;
    Fixture(int n, int k)
        : width(n), experts(k), parts(size_t(max_rows)*k*n+guard), hits(parts.n), gpu_source(parts.n),
          weights(max_rows*k), shared(max_rows*n), gate(max_rows), output(max_rows*n+guard),
          oracle_parts(parts.n), oracle_shared(shared.n), oracle_output(output.n), destinations(max_rows*k), expected(output.n) {
        ck(cudaHostAlloc(reinterpret_cast<void**>(&cpu_source),parts.n*sizeof(float),cudaHostAllocDefault),"pinned CPU source");
        ck(cudaStreamCreateWithFlags(&main,cudaStreamNonBlocking),"main stream");
        ck(cudaStreamCreateWithFlags(&producer,cudaStreamNonBlocking),"producer stream");
        ck(cudaEventCreateWithFlags(&fork,cudaEventDisableTiming),"fork event");
        ck(cudaEventCreateWithFlags(&join,cudaEventDisableTiming),"join event");
        // Warm each production entry before capture; no module allocation or
        // mutable host metadata is intentionally hidden in a captured launch.
        std::vector<float> zero(parts.n); parts.upload(zero);hits.upload(zero);
        weights.upload(std::vector<float>(weights.n,1.0f));shared.upload(std::vector<float>(shared.n));
        gate.upload(std::vector<float>(gate.n));count.upload({0});
        ck(cudaDeviceSynchronize(),"warmup inputs ready"); // uploads use the default stream; main is nonblocking
        for (int variant=0;variant<VariantCount;++variant) run_candidate(1,Variant(variant));
        ck(cudaStreamSynchronize(main),"entry warmup");
    }
    ~Fixture() {
        graphs.clear(); cudaEventDestroy(join);cudaEventDestroy(fork);
        cudaStreamDestroy(producer);cudaStreamDestroy(main);cudaFreeHost(cpu_source);
    }
    bool has_shared(Variant variant) const { return variant!=DirectNoShared; }
    void run_candidate(int rows, Variant variant) {
        if (variant==AllHits) {
            k::native_moe_combine_multi_hits(hits.p,weights.p,shared.p,output.p,width,experts,rows,main);
        } else if (variant==Gated) {
            k::native_moe_combine_multi_hits_gated(hits.p,weights.p,shared.p,gate.p,output.p,width,experts,rows,main);
        } else if (variant==Fused && k::native_moe_combine_hits_supported(parts.p,shared.p,output.p,width,experts,rows)) {
            k::native_moe_combine_multi_hits_added(parts.p,hits.p,destinations.p,count.p,weights.p,shared.p,
                                                 output.p,width,experts,rows,main);
        } else {
            if (variant==Scatter || variant==Fused)
                k::moe_hit_add(parts.p,hits.p,destinations.p,count.p,int64_t(rows)*experts,width,main);
            k::native_moe_combine_multi(parts.p,weights.p,has_shared(variant)?shared.p:nullptr,
                                       output.p,width,experts,rows,main);
        }
    }
    Graph& graph(int rows, Variant variant) {
        const auto key=std::make_pair(rows,int(variant));
        if (auto it=graphs.find(key);it!=graphs.end()) return *it->second;
        auto item=std::make_unique<Graph>();
        ck(cudaStreamBeginCapture(main,cudaStreamCaptureModeThreadLocal),"capture begin");
        ck(cudaEventRecord(fork,main),"fork record");ck(cudaStreamWaitEvent(producer,fork,0),"producer wait");
        ck(cudaMemcpyAsync(parts.p,cpu_source,parts.n*sizeof(float),cudaMemcpyHostToDevice,main),"captured CPU H2D");
        ck(cudaMemcpyAsync(hits.p,gpu_source.p,hits.n*sizeof(float),cudaMemcpyDeviceToDevice,producer),"captured GPU D2D");
        ck(cudaEventRecord(join,producer),"join record");ck(cudaStreamWaitEvent(main,join,0),"main wait");
        run_candidate(rows,variant);
        ck(cudaStreamEndCapture(main,&item->raw),"capture end");
        ck(cudaGraphInstantiate(&item->exec,item->raw,nullptr,nullptr,0),"instantiate");
        auto* ptr=item.get();graphs.emplace(key,std::move(item));return *ptr;
    }
    void prepare(const Phase& phase, int mask, Variant variant, uint32_t seed) {
        const float poison=fbits(0xffffffffu), a=fbits(0x3f800001), b=fbits(0x3f7ffffe);
        std::vector<float> cpu(parts.n,poison), gpu(parts.n,poison), effective(parts.n,poison);
        std::vector<float> ws(weights.n,poison), sh(shared.n,poison), effective_sh(shared.n,poison);
        std::vector<int32_t> dst(destinations.n,-1); int nh=0;
        for (int row=0;row<phase.rows;++row) {
            const int owner=phase.owners[row];
            for (int rank=0;rank<experts;++rank) {
                ws[row*experts+rank]=rank==0 ? (owner==1?b:1.0f) : rank==1 ? (owner==1?1.0f:b) : float(rank+1)/32.0f;
                if ((mask>>rank)&1) dst[nh++]=row*experts+rank;
            }
            for (int col=0;col<width;++col) {
                float share=float((owner*11+col*3)%17-8)/32.0f;
                if (col%8==0) share=owner==1?-1.0f:0.0f;
                if (col%8==2) share=fbits(3);
                sh[row*width+col]=share;
                effective_sh[row*width+col]=variant==Gated?om::mul_rn(share,0.5f):share;
                for (int rank=0;rank<experts;++rank) {
                    float leaf=float((owner*19+rank*13+col*7)%101-50)/16.0f;
                    if (col%8==0) leaf=rank==0?(owner==1?a:-1.0f):rank==1?(owner==1?-1.0f:a):0.0f;
                    if (col%8==2) leaf=rank==0?fbits(1):0.0f;
                    const size_t at=(size_t(row)*experts+rank)*width+col;
                    if (variant==Scatter || variant==Fused) {
                        const bool gpu_owner=(mask>>rank)&1;
                        cpu[at]=gpu_owner?0.0f:leaf;gpu[at]=gpu_owner?leaf:poison;
                        effective[at]=gpu_owner?om::add_rn(0.0f,leaf):leaf;
                    } else {
                        cpu[at]=leaf;gpu[at]=leaf;
                        effective[at]=(variant==AllHits || variant==Gated)?om::add_rn(0.0f,leaf):leaf;
                    }
                }
            }
        }
        // Atomic/arrival order is deliberately not router rank order.
        std::mt19937 rng(seed);std::shuffle(dst.begin(),dst.begin()+nh,rng);
        std::copy(cpu.begin(),cpu.end(),cpu_source);gpu_source.upload(gpu);weights.upload(ws);shared.upload(sh);
        oracle_parts.upload(effective);oracle_shared.upload(effective_sh);gate.upload(std::vector<float>(gate.n));
        destinations.upload(dst);count.upload({nh});output.poison();oracle_output.poison();
        std::fill(expected.begin(),expected.end(),poison);
        for (int row=0;row<phase.rows;++row) for (int col=0;col<width;++col) {
            const size_t base=size_t(row)*experts*width+col;
            om::Fold result(effective[base],ws[row*experts]);
            for (int rank=1;rank<experts;++rank) result.append(effective[base+size_t(rank)*width],ws[row*experts+rank]);
            expected[row*width+col]=has_shared(variant)?result.with_shared(effective_sh[row*width+col]):result.value();
        }
        std::atomic_thread_fence(std::memory_order_seq_cst);
        ck(cudaDeviceSynchronize(),"fixture ready");
    }
    void compare(const Buffer<float>& buffer, const char* label) {
        const auto got=buffer.download();
        if (std::memcmp(got.data(),expected.data(),got.size()*sizeof(float))) {
            size_t at=0;while(at<got.size() && std::bit_cast<uint32_t>(got[at])==std::bit_cast<uint32_t>(expected[at])) ++at;
            std::fprintf(stderr,"bit mismatch %s at=%zu actual=%08x expected=%08x\n",label,at,
                         std::bit_cast<uint32_t>(got[at]),std::bit_cast<uint32_t>(expected[at]));fail(label);
        }
    }
    void run() {
        const int full=(1<<experts)-1;
        std::vector<int> masks={0,full,0x155&full,0x2aa&full,1,1<<(experts-1)};
        std::sort(masks.begin(),masks.end());masks.erase(std::unique(masks.begin(),masks.end()),masks.end());
        for (size_t phase_index=0;phase_index<std::size(phases);++phase_index) {
            const auto& phase=phases[phase_index];
            for (int mask:masks) for (int v=0;v<VariantCount;++v) {
                const auto variant=Variant(v);
                std::snprintf(context,sizeof(context),"N=%d K=%d phase=%zu R=%d variant=%s mask=%x",width,experts,
                              phase_index,phase.rows,names[v],mask);
                prepare(phase,mask,variant,uint32_t(cases));
                auto& captured=graph(phase.rows,variant);
                ck(cudaGraphLaunch(captured.exec,main),"graph replay");ck(cudaStreamSynchronize(main),"graph sync");
                compare(output,"captured batched/fused/arrival result+guards");
                // Separately invoke the production scalar API per row, with
                // the identical EFFECTIVE leaves and separately rounded gate.
                for (int row=0;row<phase.rows;++row)
                    k::native_moe_combine(oracle_parts.p+size_t(row)*experts*width,weights.p+row*experts,
                        has_shared(variant)?oracle_shared.p+row*width:nullptr,oracle_output.p+row*width,width,experts,main);
                ck(cudaStreamSynchronize(main),"single row reference sync");
                compare(oracle_output,"production single row vs CPU RN fold+guards");
                if (variant==Fused && k::native_moe_combine_hits_supported(parts.p,shared.p,output.p,width,experts,phase.rows)) ++fused_calls;
                else if (variant==Fused || variant==Scatter) ++scatter_calls;
                ++cases;++graph_replays;
            }
        }
        require(graphs.size()==24,"four R buckets x six variants, independent of owner order/arrival");
        std::printf("N=%d K=%d: 24 fixed graph executables across R1/R4/R8/R12\n",width,experts);
    }
};
} // namespace
int main() {
    int devices=0;
    if (cudaGetDeviceCount(&devices)!=cudaSuccess || devices<1) {
        std::fprintf(stderr,"SKIP: NVIDIA CUDA device unavailable\n");return 77;
    }
    require(std::fesetround(FE_TONEAREST)==0,"CPU witness requires round-to-nearest");
    require(std::bit_cast<uint32_t>(om::add_rn(fbits(1),fbits(1)))==2,"CPU witness must not flush subnormals");
    size_t free=0,total=0;ck(cudaMemGetInfo(&free,&total),"memory info");
    if (free<(8ull<<30)+(64ull<<20)) {
        std::fprintf(stderr,"SKIP: preserving 8 GiB device reserve\n");return 77;
    }
    for (int width:{7,8,31,2560}) for (int experts:{1,2,10,15}) {Fixture fixture(width,experts);fixture.run();}
    require(fused_calls>0 && scatter_calls>0,"positive controls must exercise fused and R12 scatter fallback");
    require(cases==3456 && graph_replays==3456,"planned fixture/replay count");
    std::printf("PASS ordered_moe_fold_gpu: %zu fixed-leaf fixtures/replays, fused=%zu scatter=%zu; "
                "CPU RN bits == production single == captured mixed batch, inactive/guard poison intact\n",
                cases,fused_calls,scatter_calls);
    std::printf("NOT COVERED: expert Q8_K/Q8_1 equivalence, sigmoid approximation (gate=0 only), "
                "GDN/attention/state transitions, model output IDs, performance or memory qualification\n");
}
