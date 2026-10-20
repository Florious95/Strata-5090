// Asset-free reproduction of the verifier's shared-expert capture topology.
// Positive cases use the production fork-policy helper and production kernels.
// The negative case deliberately models the old merged/late missing-fork bug;
// it must reject the cross-isolation event wait, then terminate the invalid
// capture before destroying streams. This is not full-model qualification.
#include "strata/core/batch_segments.hpp"
#include "strata/kernels/elementwise.hpp"
#include "strata/kernels/verify_kernels.hpp"
#include <cuda_runtime.h>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace c = strata::core;
namespace k = strata::kernels;
namespace {
[[noreturn]] void fail(const char* label) {
    std::fprintf(stderr,"FAIL shared-stream capture: %s\n",label); std::exit(1);
}
void require(bool ok, const char* label) { if (!ok) fail(label); }
void ck(cudaError_t error, const char* label) {
    if (error != cudaSuccess) {
        std::fprintf(stderr,"%s: %d %s\n",label,int(error),cudaGetErrorString(error)); std::exit(2);
    }
}
struct Topology {
    cudaStream_t main = nullptr, shared = nullptr;
    cudaEvent_t fork = nullptr, join = nullptr;
    int32_t *source_host = nullptr, *source_device = nullptr, *output = nullptr;
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t exec = nullptr;
    Topology() {
        ck(cudaStreamCreateWithFlags(&main,cudaStreamNonBlocking),"main create");
        ck(cudaStreamCreateWithFlags(&shared,cudaStreamNonBlocking),"shared create");
        ck(cudaEventCreateWithFlags(&fork,cudaEventDisableTiming),"fork create");
        ck(cudaEventCreateWithFlags(&join,cudaEventDisableTiming),"join create");
        ck(cudaHostAlloc(reinterpret_cast<void**>(&source_host),sizeof(int32_t),cudaHostAllocMapped),"mapped source");
        ck(cudaHostGetDevicePointer(reinterpret_cast<void**>(&source_device),source_host,0),"source UVA");
        ck(cudaMalloc(reinterpret_cast<void**>(&output),2*sizeof(int32_t)),"output alloc");
        *source_host = 17;
        // Load both production kernel entry points before stream capture.
        k::copy_i32_from_mapped(output,source_device,1,shared);
        k::gpu_noop(main);
        ck(cudaStreamSynchronize(shared),"warm shared");
        ck(cudaStreamSynchronize(main),"warm main");
    }
    ~Topology() {
        if (exec) cudaGraphExecDestroy(exec);
        if (graph) cudaGraphDestroy(graph);
        cudaEventDestroy(join); cudaEventDestroy(fork);
        cudaStreamDestroy(shared); cudaStreamDestroy(main);
        cudaFree(output); cudaFreeHost(source_host);
    }
    void branch() {
        ck(cudaEventRecord(fork,main),"fork record");
        ck(cudaStreamWaitEvent(shared,fork,0),"shared joins main capture");
    }
    void shared_work() {
        k::copy_i32_from_mapped(output,source_device,1,shared);
        k::gpu_noop(shared);
        ck(cudaEventRecord(join,shared),"join record");
    }
    void capture(bool requested, bool all_resident, bool merged) {
        const bool late = c::shared_expert_fork_late(requested,all_resident,merged);
        require(late == (requested && !all_resident && !merged),"production helper truth table");
        ck(cudaStreamBeginCapture(main,cudaStreamCaptureModeThreadLocal),"positive begin");
        k::gpu_noop(main); // attention/FFN prefix
        if (!late) branch();
        k::gpu_noop(main); // router
        if (!merged) {
            k::gpu_noop(main); // per-group doorbell publication
            if (late) branch(); // only the non-merged controller emits this
        } else k::gpu_noop(main); // merged controller publication, no late fork
        shared_work();
        ck(cudaStreamWaitEvent(main,join,0),"main joins shared capture");
        // Downstream consumer must see THIS replay's shared output, not a
        // capture-time side-stream write or the preceding replay's value.
        k::copy_i32_from_mapped(output+1,output,1,main);
        k::gpu_noop(main); // combine equivalent, after the checked join
        ck(cudaStreamEndCapture(main,&graph),"positive end");
        require(graph != nullptr,"positive graph missing");
        ck(cudaGraphInstantiate(&exec,graph,nullptr,nullptr,0),"positive instantiate");
    }
    void replay(int32_t value) {
        *source_host = value;
        std::atomic_thread_fence(std::memory_order_seq_cst);
        ck(cudaMemsetAsync(output,0xa5,2*sizeof(int32_t),main),"replay poison");
        ck(cudaGraphLaunch(exec,main),"positive launch");
        ck(cudaStreamSynchronize(main),"positive sync");
        int32_t values[2] = {};
        ck(cudaMemcpy(values,output,sizeof(values),cudaMemcpyDeviceToHost),"replay output");
        require(values[0] == value,"shared work missing/stale in captured graph");
        require(values[1] == value,"downstream consumer raced shared work");
    }
};

void old_merged_late_red_case() {
    Topology bad;
    // Prime an ordinary external event, matching a previous completed
    // admission using the same verifier event handles.
    ck(cudaEventRecord(bad.join,bad.shared),"negative prime event");
    ck(cudaStreamSynchronize(bad.shared),"negative prime sync");
    ck(cudaStreamBeginCapture(bad.main,cudaStreamCaptureModeThreadLocal),"negative begin");
    k::gpu_noop(bad.main);
    constexpr bool requested = true, all_resident = false, merged = true;
    constexpr bool old_late = requested && !all_resident;
    static_assert(old_late && merged);
    // Old pre path suppresses early fork because old_late=true; the only
    // late fork lives under !merged, so it is also skipped here.
    if (!old_late) bad.branch();
    if (!merged && old_late) bad.branch();
    bad.shared_work(); // unjoined side stream: not part of bad.main's graph
    const cudaError_t wait_error = cudaStreamWaitEvent(bad.main,bad.join,0);
    const cudaError_t sticky = cudaPeekAtLastError();
    // Always end capture, including the expected invalidated case. Do not
    // launch another production kernel on the poisoned capture as a probe.
    const cudaError_t end_error = cudaStreamEndCapture(bad.main,&bad.graph);
    std::printf("expected old-policy rejection: wait=%d (%s), sticky=%d (%s), end=%d (%s)\n",
                int(wait_error),cudaGetErrorString(wait_error),int(sticky),cudaGetErrorString(sticky),
                int(end_error),cudaGetErrorString(end_error));
    cudaGetLastError(); // consume the intentional host-thread capture error
    ck(cudaStreamSynchronize(bad.shared),"negative side work cleanup");
    ck(cudaStreamSynchronize(bad.main),"negative main cleanup");
    require(wait_error != cudaSuccess,"old missing-fork external event wait unexpectedly accepted");
    require(std::strstr(cudaGetErrorString(wait_error),"captur") != nullptr,
            "negative failed for a reason other than stream capture isolation");
    require(end_error != cudaSuccess,"old invalid capture unexpectedly ended successfully");
    require(std::strstr(cudaGetErrorString(end_error),"captur") != nullptr,
            "negative capture end failed for an unrelated reason");
    require(bad.graph == nullptr,"invalidated capture returned a graph");
}
} // namespace

int main() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices < 1) {
        std::fprintf(stderr,"SKIP: NVIDIA CUDA device unavailable\n"); return 77;
    }
    // The red case is first. Every following successful capture/launch also
    // proves its invalid capture was terminated without poisoning the test.
    old_merged_late_red_case();
    int cases = 0, replays = 0;
    for (bool requested : {false,true})
        for (bool all_resident : {false,true})
            for (bool merged : {false,true}) {
                Topology good;
                good.capture(requested,all_resident,merged);
                for (int r = 0; r < 5; ++r) {
                    good.replay(1000+100*cases+r); ++replays;
                }
                std::printf("capture requested=%d all_resident=%d merged=%d late=%d replays=5\n",
                            int(requested),int(all_resident),int(merged),
                            int(c::shared_expert_fork_late(requested,all_resident,merged)));
                ++cases;
            }
    require(cases == 8 && replays == 40,"fixture count");
    std::printf("PASS shared-stream topology: 8 production-helper combinations, 40 poisoned dynamic-value "
                "graph replays, old merged-late isolation rejected and cleaned up\n");
    std::printf("NOT COVERED: full-model table/GDN/QSA/expert arithmetic, full rows12 batch, "
                "exact sampling, performance or memory qualification\n");
}
