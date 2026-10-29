// C8-6 bit contract (STRATA_IDX_POOLED_BF16, idx_pooled.hpp). Writers: from the same raw keys, a BF16 store holds
// exactly bf16_from_f32 of every FP32 store element - the prompt batch append, the per-token steps, the commit's
// layer launch - and tail, dead and block_pos are the FP32 run's bits. Readers: every scorer the BF16 mode dispatches
// gives, bit for bit, what the FP32 scorer gives on the widened keys (multi, warp, tensor-core, slots, rows, FP64);
// a graph captured in the mode keeps it. Synthetic keys, no model. Batch table paths: batch_device_qsa_gpu's BF16 arm.
#include "strata/kernels/idx_pooled.hpp"
#include "strata/kernels/native_qsa_indexer.hpp"
#include "strata/kernels/qsa.hpp"
#include "strata/kernels/qsa_select.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
namespace k = strata::kernels;
void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) { std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e)); std::exit(2); }
}
void require(bool ok, const char* what) { if (!ok) { std::fprintf(stderr, "FAIL %s\n", what); std::exit(1); } }
template <class T> struct Buffer {
    T* p = nullptr; size_t size;
    explicit Buffer(size_t n) : size(n) { ck(cudaMalloc((void**) &p, n * sizeof(T)), "alloc"); ck(cudaMemset(p, 0, n * sizeof(T)), "zero"); }
    ~Buffer() { cudaFree(p); }
    Buffer(const Buffer&) = delete;
    void upload(const std::vector<T>& v) { require(v.size() == size, "upload shape"); ck(cudaMemcpy(p, v.data(), size * sizeof(T), cudaMemcpyHostToDevice), "upload"); }
    std::vector<T> download() const { std::vector<T> v(size); ck(cudaMemcpy(v.data(), p, size * sizeof(T), cudaMemcpyDeviceToHost), "download"); return v; }
    void poison() { ck(cudaMemset(p, 0x7f, size * sizeof(T)), "poison"); }
};
template <class T> bool same_bits(const std::vector<T>& a, const std::vector<T>& b) {
    return a.size() == b.size() && !std::memcmp(a.data(), b.data(), a.size() * sizeof(T));
}
float rf(uint32_t& s) { s ^= s << 13; s ^= s >> 17; s ^= s << 5; return float(int32_t(s)) / 2147483648.0f; }
constexpr int Cells = 16384, Rows = Cells / 4 + 2, D = 128, NQ = 12, Seq = 6004, Prompt = 3001, Steps = 5000;
const k::QsaShapes S = k::qsa_real_shapes();
// one QSA layer's indexer state, sized for FP32 (a BF16 store uses the first half of `pooled`)
struct Store {
    Buffer<float> tail{3 * D}, dead{D}, pooled{size_t(Rows) * D};
    Buffer<int32_t> pos{1};
    k::QsaIndexerBuffers ib() const { return {tail.p, dead.p, pooled.p, pos.p}; }
};
// The engine's three writer paths over one sequence, the mode taken at each launch.
// raw [2][Seq][D]: the second layer of the layer launch reads raw + Seq * D.
void write(Store& st, Store& second, const float* raw, const float* gamma, const int32_t* cell, cudaStream_t stream) {
    const k::RopeScaling rope{};
    for (int p0 = 0; p0 < Prompt; p0 += 1000) {   // prefill chunks 1000, 1000, 1000, 1 (tails across chunk edges)
        const int n = std::min(1000, Prompt - p0);
        k::native_qsa_indexer_append_batch(raw + size_t(p0) * D, n, p0, 0, gamma, 1e-6f, st.ib(), S, Cells, rope, stream);
    }
    for (int p = Prompt, w = 1; p < Steps;) {   // decode windows of 1..8 rows
        const int n = std::min(w, Steps - p);
        k::native_qsa_indexer_append_steps(raw + size_t(p) * D, cell + p, 1, n, 0, gamma, 1e-6f, st.ib(), S, Cells,
                                           rope, stream);
        p += n;
        w = w % 8 + 1;
    }
    k::QsaIndexerLayers L;   // the commit's one launch over two layers
    for (int i = 0; i < 2; ++i) {
        Store& x = i ? second : st;
        L.gamma[i] = gamma; L.tail[i] = x.tail.p; L.dead[i] = x.dead.p; L.pooled[i] = x.pooled.p; L.block_pos[i] = x.pos.p;
    }
    for (int p = Steps; p < Seq; p += 4)
        k::native_qsa_indexer_append_steps_layers(raw + size_t(p) * D, int64_t(Seq) * D, cell + p, 1,
                                                  std::min(4, Seq - p), 0, 1e-6f, L, 2, S, Cells, rope, stream);
    ck(cudaStreamSynchronize(stream), "writers");
}
int main() {
#if defined(STRATA_USE_HIP) || defined(STRATA_HIP_GFX906) || defined(STRATA_GPU_METAL)
    return 77;
#endif
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices < 1) return 77;
    require(!k::qsa_idx_pooled_bf16(), "FP32 by default");
    cudaStream_t stream{};
    ck(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "stream");
    uint32_t seed = 0xC86u;
    Buffer<float> raw{size_t(2) * Seq * D}, gamma{D};
    std::vector<float> x(raw.size), g(D);
    for (auto& f : x) f = 0.2f * rf(seed);
    for (auto& f : g) f = 1.0f + 0.5f * rf(seed);
    raw.upload(x); gamma.upload(g);
    Buffer<int32_t> cell{Cells};
    std::vector<int32_t> cells(Cells);
    for (int i = 0; i < Cells; ++i) cells[i] = i;
    cell.upload(cells);
    // ---- writers: the same calls on an FP32 pair of stores (a, a2) and, in the mode, a BF16 pair (b, b2)
    Store a, a2, b, b2;
    write(a, a2, raw.p, gamma.p, cell.p, stream);
    k::qsa_set_idx_pooled_bf16(true);
    write(b, b2, raw.p, gamma.p, cell.p, stream);
    k::qsa_set_idx_pooled_bf16(false);
    int rows_checked = 0;
    for (int layer = 0; layer < 2; ++layer) {
        const Store& fa = layer ? a2 : a;
        const Store& fb = layer ? b2 : b;
        require(same_bits(fa.tail.download(), fb.tail.download()), "tail: FP32 bits in both modes");
        require(same_bits(fa.dead.download(), fb.dead.download()), "dead: FP32 bits in both modes");
        require(same_bits(fa.pos.download(), fb.pos.download()), "block_pos in both modes");
        const auto f32 = fa.pooled.download(), raw16 = fb.pooled.download();
        std::vector<uint16_t> h(size_t(Rows) * D);
        std::memcpy(h.data(), raw16.data(), h.size() * 2);
        for (size_t i = 0; i < h.size(); ++i)
            if (h[i] != k::bf16_from_f32(f32[i])) {
                std::fprintf(stderr, "FAIL pooled layer=%d row=%zu key=%zu: bf16 %04x, fp32 %08x\n", layer, i / D, i % D,
                             h[i], (unsigned) [&] { uint32_t u; std::memcpy(&u, &f32[i], 4); return u; }());
                std::exit(1);
            }
        rows_checked += Rows;
    }
    // ---- readers: the BF16 store (b) under the mode against the widened copy (w) under FP32
    const auto store16 = b.pooled.download();
    std::vector<uint16_t> h(size_t(Rows) * D);
    std::memcpy(h.data(), store16.data(), h.size() * 2);
    std::vector<float> widened(size_t(Rows) * D);
    for (size_t i = 0; i < h.size(); ++i) widened[i] = k::f32_from_bf16(h[i]);
    Buffer<float> w{size_t(Rows) * D};
    w.upload(widened);
    Buffer<float> queries{size_t(NQ) * 4 * D};
    std::vector<float> q(queries.size);
    for (auto& f : q) f = 0.1f * rf(seed);
    queries.upload(q);
    const int qpos[NQ] = {0, 1, 2, 3, 4, 7, 100, 2047, 2048, 3001, 5000, Seq - 1};
    std::vector<int32_t> st(size_t(NQ) * k::kStepCount);
    int top = 0;
    for (int i = 0; i < NQ; ++i) { k::qsa_step_fill(st.data() + i * k::kStepCount, qpos[i], S); top = std::max(top, st[i * k::kStepCount + k::kStepNBid]); }
    Buffer<int32_t> steps{st.size()};
    steps.upload(st);
    const int64_t MaxBlocks = Rows;
    Buffer<float> sb{size_t(NQ) * MaxBlocks}, sw{size_t(NQ) * MaxBlocks};
    int calls = 0;
    auto compare = [&](const char* what, auto&& call) {
        sb.poison(); sw.poison();
        k::qsa_set_idx_pooled_bf16(true);
        const bool rb = call(b.pooled.p, sb.p);
        k::qsa_set_idx_pooled_bf16(false);
        const bool rw = call(w.p, sw.p);
        ck(cudaStreamSynchronize(stream), what);
        require(rb == rw, what);
        if (rb && !same_bits(sb.download(), sw.download())) { std::fprintf(stderr, "FAIL scores %s\n", what); std::exit(1); }
        calls += rb;
    };
    compare("multi (8 queries)", [&](const float* p, float* out) {
        k::qsa_block_scores(p, b.dead.p, queries.p, steps.p, 8, MaxBlocks, S, out, stream); return true; });
    compare("warp (12 queries)", [&](const float* p, float* out) {
        k::qsa_block_scores(p, b.dead.p, queries.p, steps.p, NQ, MaxBlocks, S, out, stream); return true; });
    compare("warp (prompt, active)", [&](const float* p, float* out) {
        k::qsa_block_scores(p, b.dead.p, queries.p, steps.p, NQ, MaxBlocks, S, out, stream, top + 1); return true; });
    compare("tensor core (prompt)", [&](const float* p, float* out) {
        return k::qsa_block_scores_tc(p, b.dead.p, queries.p, steps.p, NQ, MaxBlocks, S, out, stream, top + 1); });
    compare("slots (8 sequences)", [&](const float* p, float* out) {
        const float* keys[8]; const float* dead[8];
        for (int i = 0; i < 8; ++i) { keys[i] = p; dead[i] = b.dead.p; }
        return k::qsa_block_scores_slots(keys, dead, queries.p + 4 * D * 4, steps.p + 4 * k::kStepCount, 8, MaxBlocks, S,
                                         out, stream); });
    compare("rows (8 sequences)", [&](const float* p, float* out) {
        k::QsaScoreRows r; r.n = 8;
        for (int i = 0; i < 8; ++i) { r.pooled[i] = p; r.dead[i] = b.dead.p; }
        k::qsa_block_scores_rows(r, queries.p, steps.p, 8, MaxBlocks, S, out, stream); return true; });
    for (int i : {3, 9, NQ - 1})   // the FP64 per-token path (--no-fast-select): one query's cell scores
        compare("fp64 qsa_index_step", [&](const float* p, float* out) {
            k::qsa_index_step(p, queries.p + size_t(i) * 4 * D, nullptr, S, steps.p + i * k::kStepCount,
                              st[i * k::kStepCount + k::kStepNBid] + 1, out, stream); return true; });
    require(calls >= 8, "every BF16 scorer ran");
    k::qsa_set_idx_pooled_bf16(true);
    require(!k::qsa_block_scores_rows(b.pooled.p, b.dead.p, queries.p, steps.p, 4, MaxBlocks, S, sb.p, stream),
            "the FP32-only QSAROWS kernel declines a BF16 store");
    // a graph captured in the mode keeps the BF16 kernel after the process flag changes
    cudaGraphExec_t exec{};
    {
        cudaGraph_t graph{};
        ck(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal), "capture");
        k::qsa_block_scores(b.pooled.p, b.dead.p, queries.p, steps.p, 8, MaxBlocks, S, sb.p, stream);
        ck(cudaStreamEndCapture(stream, &graph), "end capture");
        ck(cudaGraphInstantiate(&exec, graph, 0), "instantiate");
        ck(cudaGraphDestroy(graph), "graph destroy");
    }
    k::qsa_set_idx_pooled_bf16(false);
    sb.poison(); sw.poison();
    ck(cudaGraphLaunch(exec, stream), "replay");
    k::qsa_block_scores(w.p, b.dead.p, queries.p, steps.p, 8, MaxBlocks, S, sw.p, stream);
    ck(cudaStreamSynchronize(stream), "replay sync");
    require(same_bits(sb.download(), sw.download()), "captured BF16 graph after the flag changes");
    ck(cudaGraphExecDestroy(exec), "exec destroy");
    // ---- the rounding is real: FP32 keys score differently somewhere (informational: synthetic selection overlap)
    sb.poison(); sw.poison();
    k::qsa_block_scores(a.pooled.p, a.dead.p, queries.p, steps.p, 8, MaxBlocks, S, sb.p, stream);
    k::qsa_block_scores(w.p, b.dead.p, queries.p, steps.p, 8, MaxBlocks, S, sw.p, stream);
    ck(cudaStreamSynchronize(stream), "fp32 vs bf16");
    const auto s32 = sb.download(), s16 = sw.download();
    size_t differ = 0;
    for (int i = 0; i < 8; ++i)
        for (int blk = 0; blk <= st[i * k::kStepCount + k::kStepNBid]; ++blk)
            differ += std::memcmp(&s32[size_t(i) * MaxBlocks + blk], &s16[size_t(i) * MaxBlocks + blk], 4) != 0;
    require(differ > 0, "BF16 keys change some score");
    ck(cudaStreamDestroy(stream), "stream destroy");
    std::printf("idx_pooled_gpu: PASS writers=batch,steps,layers rows=%d bits=bf16(fp32); readers=%d calls bitwise vs "
                "widened FP32 (multi,warp,active,tc,slots,rows,fp64); captured graph keeps BF16; %zu of the FP32 "
                "scores move\n", rows_checked, calls, differ);
}
