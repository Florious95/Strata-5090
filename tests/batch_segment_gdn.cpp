#include "strata/kernels/verify_kernels.hpp"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static char case_info[128] = {};

static void check(cudaError_t e) {
    if (e != cudaSuccess) { std::fprintf(stderr, "%s\n", cudaGetErrorString(e)); std::exit(1); }
}
struct Buffer {
    float* p = nullptr;
    size_t n;
    explicit Buffer(size_t count): n(count) { check(cudaMalloc((void**)&p, n * 4)); }
    ~Buffer() { cudaFree(p); }
    void fill(float scale, int seed) {
        std::vector<float> h(n);
        for (size_t i = 0; i < n; ++i) h[i] = scale * (float)((int)((i * 17 + seed) % 101) - 50);
        check(cudaMemcpy(p, h.data(), n * 4, cudaMemcpyHostToDevice));
    }
    void copy(const Buffer& src) { check(cudaMemcpy(p, src.p, n * 4, cudaMemcpyDeviceToDevice)); }
    void equal(const Buffer& other, const char* label) const {
        std::vector<float> a(n), b(n);
        check(cudaMemcpy(a.data(), p, n * 4, cudaMemcpyDeviceToHost));
        check(cudaMemcpy(b.data(), other.p, n * 4, cudaMemcpyDeviceToHost));
        if (std::memcmp(a.data(), b.data(), n * 4)) {
            size_t i = 0;
            while (!std::memcmp(a.data() + i, b.data() + i, 4)) ++i;
            uint32_t av = 0, bv = 0;
            std::memcpy(&av, a.data() + i, 4); std::memcpy(&bv, b.data() + i, 4);
            std::fprintf(stderr, "FAIL: %s (%s), element=%zu actual=%08x reference=%08x\n",
                         label, case_info, i, av, bv);
            std::exit(1);
        }
    }
};

int main() {
    using namespace strata::kernels;
    size_t free = 0, total = 0;
    check(cudaMemGetInfo(&free, &total));
    if (free < (8ull << 30) + (32ull << 20)) {
        std::fprintf(stderr, "SKIP: preserving 8 GiB reserve\n"); return 77;
    }
    constexpr int HK = 2, HV = 4, C = (2 * HK + HV) * 128, Z = HV * 128, ST = 128 * HV * 128;
    Buffer initial(2 * ST), history(2 * C * 3), state(2 * ST), hist(2 * C * 3),
           reference(2 * ST), refhist(2 * C * 3), weights(C * 4), gamma(128),
           qkv(12 * C), h(12 * C), gate(12 * HV), beta(12 * HV), z(12 * Z), y(12 * Z), refy(12 * Z), scratch(C);
    initial.fill(0.001f, 1); history.fill(0.005f, 3); weights.fill(0.002f, 7);
    gamma.fill(0.01f, 11); qkv.fill(0.003f, 13); gate.fill(0.001f, 17);
    beta.fill(0.001f, 19); z.fill(0.01f, 23);
    int32_t* keep = nullptr;
    check(cudaMalloc((void**)&keep, 8));
    auto commit_count = [&](int k, int slot = 0) { check(cudaMemcpy(keep + slot, &k, 4, cudaMemcpyHostToDevice)); };
    cudaStream_t main_stream = nullptr, side_stream = nullptr;
    cudaEvent_t fork = nullptr, join = nullptr, initialized = nullptr;
    check(cudaStreamCreateWithFlags(&main_stream, cudaStreamNonBlocking));
    check(cudaStreamCreateWithFlags(&side_stream, cudaStreamNonBlocking));
    check(cudaEventCreateWithFlags(&fork, cudaEventDisableTiming));
    check(cudaEventCreateWithFlags(&join, cudaEventDisableTiming));
    check(cudaEventCreateWithFlags(&initialized, cudaEventDisableTiming));
    // Exercise the actual fork/join capture dependencies with two private
    // slot states, not merely sequential kernels bearing a parallel label.
    auto execute = [&](bool parallel, auto launch) {
        // Fixture resets/count uploads use the default stream. D2D cudaMemcpy
        // and cudaMemset may still be pending when their host calls return;
        // nonblocking streams do not implicitly wait for them. Drain this
        // explicit fixture-ready event before capture/replay, not a model
        // hot-path device synchronization or a change to kernel arithmetic.
        check(cudaEventRecord(initialized, nullptr));
        check(cudaEventSynchronize(initialized));
        check(cudaStreamBeginCapture(main_stream, cudaStreamCaptureModeThreadLocal));
        if (parallel) {
            check(cudaEventRecord(fork, main_stream));
            check(cudaStreamWaitEvent(side_stream, fork, 0));
        }
        launch(0, main_stream);
        launch(1, parallel ? side_stream : main_stream);
        if (parallel) {
            check(cudaEventRecord(join, side_stream));
            check(cudaStreamWaitEvent(main_stream, join, 0));
        }
        cudaGraph_t graph = nullptr;
        cudaGraphExec_t exec = nullptr;
        check(cudaStreamEndCapture(main_stream, &graph));
        check(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
        check(cudaGraphLaunch(exec, main_stream));
        check(cudaStreamSynchronize(main_stream));
        check(cudaGraphExecDestroy(exec));
        check(cudaGraphDestroy(graph));
    };
    auto sequential = [&](int a, int b, int ka, int kb) {
        reference.copy(initial); refhist.copy(history);
        int offsets[] = {0, a}, counts[] = {ka, kb};
        commit_count(1);
        for (int slot = 0; slot < 2; ++slot) for (int j = 0; j < counts[slot]; ++j) {
            const int row = offsets[slot] + j;
            gdn_conv_l2_multi(refhist.p + slot * C * 3, qkv.p + row * C, weights.p, scratch.p, C, 2 * HK, 1e-6f, 1, nullptr);
            gdn_step_norm_multi(reference.p + slot * ST, scratch.p, C, gate.p + row * HV, beta.p + row * HV,
                                z.p + row * Z, gamma.p, 1e-6f, refy.p + row * Z, HK, HV, 1, keep, nullptr);
            gdn_conv_commit(refhist.p + slot * C * 3, qkv.p + row * C, C, keep, nullptr);
        }
        check(cudaDeviceSynchronize());
    };
    int cases = 0;
    for (bool parallel : {false, true})
    for (int a = 1; a <= 6; ++a) for (int b = 1; b <= 6; ++b) {
        std::snprintf(case_info, sizeof case_info, "mode=%s widths=%d,%d phase=verify", parallel ? "parallel" : "serial", a, b);
        state.copy(initial); hist.copy(history);
        check(cudaMemset(y.p, 0, y.n * 4)); check(cudaMemset(refy.p, 0, refy.n * 4));
        int offsets[] = {0, a}, counts[] = {a, b};
        execute(parallel, [&](int slot, cudaStream_t stream) {
            const int row = offsets[slot], width = counts[slot];
            gdn_conv_l2_multi(hist.p + slot * C * 3, qkv.p + row * C, weights.p, h.p + row * C, C, 2 * HK, 1e-6f, width, stream);
            gdn_step_norm_multi(state.p + slot * ST, h.p + row * C, C, gate.p + row * HV, beta.p + row * HV,
                                z.p + row * Z, gamma.p, 1e-6f, y.p + row * Z, HK, HV, width, nullptr, stream);
        });
        state.equal(initial, "verification mutated recurrent state"); hist.equal(history, "verification mutated convolution history");
        sequential(a, b, a, b);
        y.equal(refy, "segmented outputs differ from sequential outputs");
        for (int ka = 1; ka <= a; ++ka) for (int kb = 1; kb <= b; ++kb) {
            std::snprintf(case_info, sizeof case_info, "mode=%s widths=%d,%d keep=%d,%d phase=commit",
                          parallel ? "parallel" : "serial", a, b, ka, kb);
            state.copy(initial); hist.copy(history);
            int accepted[] = {ka, kb};
            for (int slot = 0; slot < 2; ++slot) commit_count(accepted[slot], slot);
            execute(parallel, [&](int slot, cudaStream_t stream) {
                const int row = offsets[slot];
                gdn_conv_commit(hist.p + slot * C * 3, qkv.p + row * C, C, keep + slot, stream);
                gdn_step_norm_multi(state.p + slot * ST, h.p + row * C, C, gate.p + row * HV, beta.p + row * HV,
                                    z.p + row * Z, gamma.p, 1e-6f, y.p + row * Z, HK, HV, counts[slot], keep + slot, stream);
            });
            sequential(a, b, ka, kb);
            state.equal(reference, "partial commit recurrent state"); hist.equal(refhist, "partial commit convolution history");
            ++cases;
        }
    }
    check(cudaFree(keep));
    check(cudaEventDestroy(initialized));
    check(cudaEventDestroy(join)); check(cudaEventDestroy(fork));
    check(cudaStreamDestroy(side_stream)); check(cudaStreamDestroy(main_stream));
    std::printf("PASS: 72 captured two-slot layouts (serial/parallel, up to 6+6), %d partial commits, bitwise outputs and state\n", cases);
}
