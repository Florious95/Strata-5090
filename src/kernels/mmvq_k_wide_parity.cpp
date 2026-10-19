// Synthetic GPU contract for the actual wide-head composition. No model/fixtures.
// Tests every 9..12-column specialization, canonical single-column and old split
// oracles, finite outputs, odd row tails, changed-input poisoned graph replays,
// and a non-exact negative control. A graph must have TWO quantizers and ONE head.
#include "strata/kernels/native_mmvq.hpp"
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
namespace k = strata::kernels;
void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}
void require(bool ok, const char* what) { if (!ok) throw std::runtime_error(what); }
template<class T> T* alloc(size_t n) {
    T* p = nullptr;
    ck(cudaMalloc(reinterpret_cast<void**>(&p), n * sizeof(T)), "malloc");
    return p;
}
std::vector<uint32_t> read_bits(const float* p, size_t n) {
    std::vector<uint32_t> h(n);
    ck(cudaMemcpy(h.data(), p, n * 4, cudaMemcpyDeviceToHost), "read output");
    return h;
}
size_t differences(const std::vector<uint32_t>& a, const std::vector<uint32_t>& b) {
    size_t diff = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        float fa, fb;
        std::memcpy(&fa, &a[i], 4); std::memcpy(&fb, &b[i], 4);
        require(std::isfinite(fa) && std::isfinite(fb), "non-finite output");
        diff += a[i] != b[i];
    }
    return diff;
}
struct Counts { size_t direct = 0, graph = 0, values = 0, controls = 0; } all_counts[2][4];

void run(int type, int n_in, int n_out, cudaStream_t s) {
    auto* counts = all_counts[type - 13];
    const size_t wb = k::native_mmvq_weight_bytes(type, n_in, n_out);
    const size_t qcol = k::native_q8_1_bytes(n_in, 1);
    std::mt19937 rng(1524u + unsigned(type) + unsigned(n_in) + unsigned(n_out));
    std::vector<uint8_t> w(wb);
    // Four random bytes per PRNG call, then replace every FP16 scale with a
    // finite signed normal. Q5 packed scale/min bits and Q6 signed scales stay random.
    for (size_t i = 0; i < wb; i += 4) {
        const uint32_t v = rng();
        std::memcpy(w.data() + i, &v, std::min(size_t(4), wb - i));
    }
    const size_t block_bytes = type == 13 ? 176 : 210;
    for (size_t i = 0; i < wb; i += block_bytes)
        for (int j = 0; j < (type == 13 ? 2 : 1); ++j) {
            const uint32_t v = rng();
            const uint16_t d = uint16_t(((v >> 31) << 15) | ((5u + (v >> 10) % 5u) << 10) | (v & 1023u));
            std::memcpy(w.data() + i + (type == 13 ? size_t(j * 2) : 208), &d, 2);
        }
    auto* dw = alloc<uint8_t>(wb);
    auto* dx = alloc<float>(12 * size_t(n_in));
    auto* qwide = alloc<uint8_t>(12 * qcol);
    auto* qref = alloc<uint8_t>(12 * qcol);
    auto* il = alloc<uint8_t>(k::native_q8_1_il_bytes(n_in, 4));
    const size_t capacity = 12 * size_t(n_out);
    auto* dy = alloc<float>(capacity);
    auto* dr = alloc<float>(capacity);
    ck(cudaMemcpy(dw, w.data(), wb, cudaMemcpyHostToDevice), "copy weights");

    // Unsupported widths and rounding mode must enqueue nothing, not silently
    // fall back to a different reduction. Null pointers are safe in this case.
    require(!k::native_head_f32_wide(type, nullptr, nullptr, nullptr, nullptr, n_in, n_out, 8, s), "accepted T=8");
    require(!k::native_head_f32_wide(type, nullptr, nullptr, nullptr, nullptr, n_in, n_out, 13, s), "accepted T=13");
    require(!k::native_head_f32_wide(12, nullptr, nullptr, nullptr, nullptr, n_in, n_out, 12, s), "accepted Q4_K");
    k::native_mmvq_set_multi_exact(false);
    require(!k::native_head_f32_wide(type, nullptr, nullptr, nullptr, nullptr, n_in, n_out, 12, s), "accepted non-exact mode");
    k::native_mmvq_set_multi_exact(true);

    for (int T = 9; T <= 12; ++T) {
        const size_t ny = size_t(T) * n_out;
        const size_t yb = ny * 4;
        // This is the public composition used by Verifier, not a test-only kernel.
        ck(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal), "begin capture");
        require(k::native_head_f32_wide(type, dw, dx, qwide, dy, n_in, n_out, T, s), "wide capture declined");
        cudaGraph_t graph = nullptr;
        ck(cudaStreamEndCapture(s, &graph), "end capture");
        size_t nn = 0;
        ck(cudaGraphGetNodes(graph, nullptr, &nn), "node count");
        require(nn == 3, "wide graph is not two quantizers plus one projection");
        std::vector<cudaGraphNode_t> nodes(nn);
        ck(cudaGraphGetNodes(graph, nodes.data(), &nn), "nodes");
        for (auto node : nodes) {
            cudaGraphNodeType type;
            ck(cudaGraphNodeGetType(node, &type), "node type");
            require(type == cudaGraphNodeTypeKernel, "non-kernel in wide graph");
        }
        cudaGraphExec_t exec = nullptr;
        ck(cudaGraphInstantiate(&exec, graph, 0), "instantiate");
        ck(cudaGraphDestroy(graph), "destroy graph");

        for (int replay = 0; replay < 2; ++replay) {
            std::vector<float> x(size_t(T) * n_in);
            // Normal amplitudes, then alternating-sign large cancellation input.
            // Column-dependent values catch stride mistakes and stale graph data.
            for (size_t i = 0; i < x.size(); ++i) {
                const float v = float(int(rng() % 20001u) - 10000) / 10000.f;
                x[i] = replay ? ((i & 1) ? -v : v) * 31.f : v * float(1 + i / n_in);
            }
            ck(cudaMemcpyAsync(dx, x.data(), x.size() * 4, cudaMemcpyHostToDevice, s), "copy input");
            // Independent <=8-column quantization oracle, contiguous Q8 columns.
            for (int first = 0; first < T; first += 8)
                k::native_quantize_q8_1(dx + size_t(first) * n_in, qref + size_t(first) * qcol,
                                        n_in, std::min(8, T - first), s);
            for (int j = 0; j < T; ++j)
                k::native_mmvq(type, dw, qref + size_t(j) * qcol, dr + size_t(j) * n_out, n_in, n_out, 1, s);
            ck(cudaStreamSynchronize(s), "single oracle sync");
            const auto single = read_bits(dr, ny);

            // Original 8+(T-8) dispatch, including the 2..4-column IL head path.
            k::native_mmvq(type, dw, qref, dr, n_in, n_out, 8, s);
            const int tail = T - 8;
            if (k::native_mmvq_il_supported(type, tail, n_out)) {
                k::native_q8_1_interleave(qref + 8 * qcol, il, n_in, tail, s);
                k::native_mmvq_il(type, dw, qref + 8 * qcol, il, dr + 8 * size_t(n_out), n_in, n_out, tail, s);
            } else
                k::native_mmvq(type, dw, qref + 8 * qcol, dr + 8 * size_t(n_out), n_in, n_out, tail, s);
            ck(cudaStreamSynchronize(s), "split oracle sync");
            require(differences(read_bits(dr, ny), single) == 0, "split/single mismatch");

            ck(cudaMemsetAsync(dy, 0xff, yb, s), "poison direct output");
            ck(cudaMemsetAsync(qwide, 0xa5, size_t(T) * qcol, s), "poison direct Q8");
            require(k::native_head_f32_wide(type, dw, dx, qwide, dy, n_in, n_out, T, s), "wide direct declined");
            ck(cudaStreamSynchronize(s), "wide direct sync");
            require(differences(read_bits(dy, ny), single) == 0, "wide direct/single mismatch");
            ++counts[T - 9].direct;

            ck(cudaMemsetAsync(dy, 0xff, yb, s), "poison graph output");
            ck(cudaMemsetAsync(qwide, 0xa5, size_t(T) * qcol, s), "poison graph Q8");
            ck(cudaGraphLaunch(exec, s), "replay");
            ck(cudaStreamSynchronize(s), "wide graph sync");
            require(differences(read_bits(dy, ny), single) == 0, "wide graph/single mismatch");
            ++counts[T - 9].graph;
            counts[T - 9].values += 2 * ny;

            // Different warp layout MUST differ on long reductions; otherwise a
            // test that accidentally accepted rounding differences has no power.
            if (n_in >= 2560 && replay == 0) {
                k::native_mmvq_set_multi_exact(false);
                k::native_mmvq(type, dw, qref, dr, n_in, n_out, 8, s);
                ck(cudaStreamSynchronize(s), "negative control sync");
                require(differences(read_bits(dr, size_t(8) * n_out),
                                     std::vector<uint32_t>(single.begin(), single.begin() + size_t(8) * n_out)) > 0,
                        "negative control has no power");
                k::native_mmvq_set_multi_exact(true);
                ++counts[T - 9].controls;
            }
        }
        std::printf("type=%d %d x %d T=%d: direct=2 graph=2 nodes=3 diffs=0 nonfinite=0 PASS\n", type, n_in, n_out, T);
        ck(cudaGraphExecDestroy(exec), "destroy exec");
    }
    cudaFree(dr); cudaFree(dy); cudaFree(il); cudaFree(qref); cudaFree(qwide); cudaFree(dx); cudaFree(dw);
}
} // namespace

int main() {
    try {
        int ndev = 0;
        ck(cudaGetDeviceCount(&ndev), "GPU inventory");
        require(ndev > 0, "GPU contract requires a real CUDA device");
        cudaDeviceProp p{};
        ck(cudaGetDeviceProperties(&p, 0), "device properties");
        std::printf("mmvq_k_wide_parity GPU=%s sm_%d%d\n", p.name, p.major, p.minor);
        cudaStream_t s = nullptr;
        ck(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking), "stream");
        for (int type : {13, 14}) {
            run(type, 256, 1, s);
            run(type, 512, 3, s);
            run(type, 2560, 257, s);
            run(type, 4096, 513, s);
            run(type, 2560, 248320, s); // Real frozen head is type13, 437043200 bytes.
            for (int T = 9; T <= 12; ++T) {
                const auto& c = all_counts[type - 13][T - 9];
                std::printf("TOTAL type=%d T=%d direct=%zu graph=%zu compared_values=%zu negative_controls=%zu diffs=0 nonfinite=0\n",
                            type, T, c.direct, c.graph, c.values, c.controls);
            }
        }
        ck(cudaStreamDestroy(s), "destroy stream");
        std::printf("mmvq_k_wide_parity: PASS\n");
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "mmvq_k_wide_parity: FAIL: %s\n", e.what());
        return 1;
    }
}
