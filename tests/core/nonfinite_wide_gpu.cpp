// Actual v43 production scan + mapped host consumer. Synthetic logits; no model.
#include "strata/kernels/verify_kernels.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace k = strata::kernels;
static void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) { std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e)); std::exit(2); }
}
int main() {
    constexpr uint32_t canary = 0x9ae35712u;
    constexpr int guard = 16;
    int launches[17] = {}, cases = 0;
    for (int rows : {1, 8, 9, 10, 11, 12, 16}) for (int vocab : {17, 248320, 248321}) {
        // Exact logical capacity, with canaries immediately before/after all active flags.
        const int words = rows * k::kNonfiniteBlocks;
        uint32_t *host = nullptr, *mapped = nullptr;
        float* logits = nullptr;
        ck(cudaHostAlloc(&host, (words + 2 * guard) * sizeof(uint32_t), cudaHostAllocMapped), "host alloc");
        ck(cudaHostGetDevicePointer(&mapped, host, 0), "host alias");
        ck(cudaMalloc(&logits, size_t(rows) * vocab * sizeof(float)), "logits alloc");
        cudaStream_t stream{};
        ck(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "stream");
        cudaGraph_t graph{}; cudaGraphExec_t exec{};
        ck(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal), "capture");
        k::logits_nonfinite_rows(logits, rows, vocab, mapped + guard, stream);
        ck(cudaStreamEndCapture(stream, &graph), "end capture");
        ck(cudaGraphInstantiate(&exec, graph, 0), "instantiate");
        std::vector<float> input(size_t(rows) * vocab, 0.125f);
        auto run = [&](uint32_t expected, bool replay) {
            std::fill(host, host + words + 2 * guard, canary);
            ck(cudaMemcpyAsync(logits, input.data(), input.size() * sizeof(float), cudaMemcpyHostToDevice, stream), "upload");
            if (replay) ck(cudaGraphLaunch(exec, stream), "replay");
            else k::logits_nonfinite_rows(logits, rows, vocab, mapped + guard, stream);
            ck(cudaStreamSynchronize(stream), "sync");
            if (k::nonfinite_row_mask(host + guard, rows, rows) != expected) {
                std::fprintf(stderr, "mask mismatch rows=%d vocab=%d expected=%x replay=%d\n", rows, vocab, expected, replay); std::exit(1);
            }
            for (int i = 0; i < guard; ++i)
                if (host[i] != canary || host[guard + words + i] != canary) { std::fprintf(stderr, "canary write\n"); std::exit(1); }
            for (int i = 0; i < words; ++i)
                if (host[guard + i] > 1) { std::fprintf(stderr, "stale/unwritten flag\n"); std::exit(1); }
            ++launches[rows];
        };
        for (bool replay : {false, true}) {
            run(0, replay); // Stale poison must be overwritten.
            for (int row = 0; row < rows; ++row) for (uint32_t bits : {0x7fc00001u, 0x7f800000u, 0xff800000u}) {
                for (int at : {0, vocab / 2, vocab - 1}) {
                    const size_t i = size_t(row) * vocab + at;
                    std::memcpy(&input[i], &bits, sizeof(bits));
                    run(1u << row, replay);
                    input[i] = 0.125f;
                    ++cases;
                }
            }
            // Finite subnormals and largest finite values must not false-positive.
            const uint32_t finite[] = {1u, 0x7f7fffffu, 0xff7fffffu};
            for (int row = 0; row < rows; ++row) std::memcpy(&input[size_t(row) * vocab], &finite[row % 3], sizeof(float));
            run(0, replay);
            std::fill(input.begin(), input.end(), 0.125f);
        }
        ck(cudaGraphExecDestroy(exec), "destroy exec"); ck(cudaGraphDestroy(graph), "destroy graph");
        ck(cudaStreamDestroy(stream), "destroy stream"); ck(cudaFree(logits), "free logits"); ck(cudaFreeHost(host), "free host");
    }
    std::printf("PASS: %d NaN/+Inf/-Inf row/position direct+graph cases; exact mapped bounds; no stale flags; finite edges\n", cases);
    for (int rows : {1, 8, 9, 10, 11, 12, 16}) std::printf("rows=%d launches=%d\n", rows, launches[rows]);
}
