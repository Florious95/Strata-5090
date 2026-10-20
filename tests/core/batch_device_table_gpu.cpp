// Public, asset-free parity test for the production device-table GDN/PLE APIs.
// This is NOT full-model or QSA qualification: no Q4 append/indexer/selection/
// union/attention path is executed here. The table intentionally contains only
// target owners, so validate_device_slot_table (which requires all 48 QSA
// entries) must not be called on this operator-scoped fixture.
#include "strata/core/batch_device_layout.hpp"
#include "strata/kernels/batch_device_ops.hpp"
#include "strata/kernels/elementwise.hpp"
#include "strata/kernels/ngram.hpp"
#include "strata/kernels/ple.hpp"
#include "strata/kernels/verify_kernels.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace k = strata::kernels;
namespace c = strata::core;
namespace {
char context[192] = {};
size_t comparisons = 0, cases = 0, rejected_commits = 0;
[[noreturn]] void fail(const char* label) {
    std::fprintf(stderr, "FAIL: %s (%s)\n", label, context); std::exit(1);
}
void require(bool ok, const char* label) { if (!ok) fail(label); }
void ck(cudaError_t e, const char* label) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "%s: %s (%s)\n", label, cudaGetErrorString(e), context); std::exit(2);
    }
}
template<class T> struct Buffer {
    T* p = nullptr;
    size_t n;
    explicit Buffer(size_t size) : n(size) { ck(cudaMalloc(&p, n * sizeof(T)), "cudaMalloc"); }
    ~Buffer() { cudaFree(p); }
    Buffer(const Buffer&) = delete;
    void upload(const std::vector<T>& values) {
        require(values.size() == n, "upload size");
        ck(cudaMemcpy(p, values.data(), n * sizeof(T), cudaMemcpyHostToDevice), "upload");
    }
    std::vector<T> download() const {
        std::vector<T> values(n);
        ck(cudaMemcpy(values.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost), "download"); return values;
    }
    void poison() { ck(cudaMemset(p, 0xff, n * sizeof(T)), "poison"); }
};
template<class T> struct Pinned {
    T *host = nullptr, *device = nullptr;
    Pinned() {
        ck(cudaHostAlloc(reinterpret_cast<void**>(&host), sizeof(T), cudaHostAllocMapped), "pinned alloc");
        ck(cudaHostGetDevicePointer(reinterpret_cast<void**>(&device), host, 0), "mapped pointer"); *host = {};
    }
    ~Pinned() { cudaFreeHost(host); }
    Pinned(const Pinned&) = delete;
};
template<class T> void equal(const std::vector<T>& a, const std::vector<T>& b, const char* label) {
    require(a.size() == b.size(), "comparison size"); ++comparisons;
    if (!std::memcmp(a.data(), b.data(), a.size() * sizeof(T))) return;
    size_t i = 0;
    while (i < a.size() && !std::memcmp(&a[i], &b[i], sizeof(T))) ++i;
    std::fprintf(stderr, "bit mismatch: %s element=%zu (%s)\n", label, i, context); std::exit(1);
}
void equal(const Buffer<float>& a, const Buffer<float>& b, const char* label) {
    equal(a.download(), b.download(), label);
}
float noise(uint32_t& seed) {
    seed ^= seed << 13; seed ^= seed >> 17; seed ^= seed << 5;
    return float(int32_t(seed)) / 2147483648.0f;
}
void fill(Buffer<float>& buffer, uint32_t seed, float scale, float offset = 0.0f) {
    std::vector<float> values(buffer.n);
    for (auto& x : values) x = offset + scale * noise(seed);
    buffer.upload(values);
}
struct Layout {
    const char* name;
    int cut;
    std::vector<std::pair<int, int>> segments; // owner, length, in causal row order
};
const std::vector<Layout> layouts = {
    {"6+6 owners0/3", 6, {{0,6},{3,6}}},
    {"3+3+3+3 owners3/1/0/2", 6, {{3,3},{1,3},{0,3},{2,3}}},
    {"6+6 owners2/1", 6, {{2,6},{1,6}}},
    {"3+3+3+3 owners1/0/2/3", 6, {{1,3},{0,3},{2,3},{3,3}}},
    {"4+4+4 owners0/3/1", 8, {{0,4},{3,4},{1,4}}},
    {"8+4 owners3/0", 8, {{3,8},{0,4}}},
    {"4+4+4 owners2/0/3", 8, {{2,4},{0,4},{3,4}}},
    {"8+4 owners1/2", 8, {{1,8},{2,4}}},
    {"1+5+6 owners1/3/2", 6, {{1,1},{3,5},{2,6}}},
    {"1+2+5+4 owners0/2/1/3", 8, {{0,1},{2,2},{1,5},{3,4}}},
    {"solo6 owner3", 0, {{3,6}}},
    {"solo6 owner0", 0, {{0,6}}},
    {"solo8 owner2", 0, {{2,8}}},
    {"solo8 owner1", 0, {{1,8}}},
};
struct Graph {
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t exec = nullptr;
    size_t nodes = 0, launches = 0;
    ~Graph() { if (exec) cudaGraphExecDestroy(exec); if (graph) cudaGraphDestroy(graph); }
};

struct Fixture {
    static constexpr int rows = 12, layers = 2, guard = 128;
    static constexpr int hist_floats = k::NG_HIST * k::NG_HC_DIM;
    static constexpr uint64_t table_epoch = 997;
    const int hk, hv, channels, value_dim;
    const int64_t state_floats, conv_offset, state_stride;
    const k::BatchGdnShape shape;
    std::array<std::unique_ptr<Buffer<float>>, 4> reference, candidate, ref_ple, new_ple;
    std::array<std::vector<float>, 4> initial, ple_initial;
    Buffer<float> weights, gamma, qkv, gate, beta, z, normalized;
    Buffer<float> ref_h, new_h, ref_y, new_y, ref_snap, new_snap;
    Buffer<int32_t> keeps{4};
    Buffer<k::DeviceSlotTable> table{1};
    Buffer<k::DeviceWindowLayout> window{1};
    Buffer<k::DeviceCommitLayout> commit{1};
    Buffer<uint64_t> epoch{1};
    Pinned<k::DeviceWindowLayout> staged_window;
    Pinned<k::DeviceCommitLayout> staged_commit;
    Pinned<uint32_t> status;
    k::DeviceBatchView view{};
    cudaStream_t main = nullptr, side = nullptr;
    cudaEvent_t fork = nullptr, join = nullptr;
    std::map<std::pair<int,int>, std::unique_ptr<Graph>> targets;
    std::unique_ptr<Graph> commit_graph;
    uint64_t sequence = 0;
    std::array<uint64_t, 4> owner_epochs{101,103,107,109};
    size_t target_launches = 0;

    Fixture(int key_heads, int value_heads)
        : hk(key_heads), hv(value_heads), channels((2*hk+hv)*128), value_dim(hv*128),
          state_floats(int64_t(128)*hv*128), conv_offset(state_floats + guard),
          state_stride(conv_offset + 3*channels + guard),
          shape{channels,hk,hv,state_stride,conv_offset},
          weights(layers*channels*4), gamma(layers*128), qkv(layers*rows*channels),
          gate(layers*rows*hv), beta(layers*rows*hv), z(layers*rows*value_dim),
          normalized(rows*k::NG_HC_DIM), ref_h(layers*rows*channels+guard),
          new_h(layers*rows*channels+guard), ref_y(layers*rows*value_dim+guard),
          new_y(layers*rows*value_dim+guard), ref_snap(rows*hist_floats+guard),
          new_snap(rows*hist_floats+guard) {
        ck(cudaStreamCreateWithFlags(&main, cudaStreamNonBlocking), "main stream");
        ck(cudaStreamCreateWithFlags(&side, cudaStreamNonBlocking), "side stream");
        ck(cudaEventCreateWithFlags(&fork, cudaEventDisableTiming), "fork event");
        ck(cudaEventCreateWithFlags(&join, cudaEventDisableTiming), "join event");
        k::DeviceSlotTable host_table{};
        for (int b = 0; b < 4; ++b) {
            const size_t n = 2*guard + layers*state_stride;
            reference[b] = std::make_unique<Buffer<float>>(n);
            candidate[b] = std::make_unique<Buffer<float>>(n);
            ref_ple[b] = std::make_unique<Buffer<float>>(2*guard+hist_floats);
            new_ple[b] = std::make_unique<Buffer<float>>(2*guard+hist_floats);
            auto& owner = host_table.target[b];
            owner.gdn_base = candidate[b]->p + guard;
            owner.ple_hist = new_ple[b]->p + guard;
            owner.max_cells = 262144; owner.address_epoch = owner_epochs[b];
        }
        table.upload({host_table}); epoch.upload({table_epoch});
        view = {table.p,window.p,commit.p,epoch.p,status.device};
    }
    ~Fixture() {
        // Captured arguments and staging outlive every graph using them.
        targets.clear(); commit_graph.reset();
        cudaEventDestroy(join); cudaEventDestroy(fork);
        cudaStreamDestroy(side); cudaStreamDestroy(main);
    }
    float* state(int b, int gi) { return reference[b]->p+guard+gi*state_stride; }
    float* history(int b) { return ref_ple[b]->p+guard; }
    void reset(const Layout& layout, int prefix_pattern) {
        int ids[rows] = {}, accepted[4] = {}; int64_t positions[rows] = {};
        bool active[4] = {}; int n = 0, segment_index = 0;
        for (auto [owner, length] : layout.segments) {
            active[owner] = true;
            int keep = 1;
            if (prefix_pattern == 1) keep = 1 + ((owner+segment_index+int(sequence)) % length);
            if (prefix_pattern == 2) keep = length;
            if (prefix_pattern == 3) keep = std::max(1,length-1);
            accepted[owner] = keep; ++segment_index;
            for (int j = 0; j < length; ++j, ++n) {
                ids[n] = owner;
                positions[n] = 262120 - 11*owner + j;
            }
        }
        ++sequence;
        std::array<uint64_t,4> generations{sequence+11,sequence+23,sequence+37,sequence+53};
        std::string error;
        require(c::make_device_window(ids,positions,n,layout.cut,generations,owner_epochs,
                                     sequence,table_epoch,17,*staged_window.host,error), error.c_str());
        require(c::make_device_commit(*staged_window.host,accepted,*staged_commit.host,error), error.c_str());
        keeps.upload({accepted[0],accepted[1],accepted[2],accepted[3]});
        // Inactive mutable arenas contain NaN poison, rather than plausible
        // zeros. Guard bands include the gap before/after each layer's history.
        for (int b = 0; b < 4; ++b) {
            initial[b].resize(reference[b]->n);
            ple_initial[b].resize(ref_ple[b]->n);
            std::memset(initial[b].data(),0xff,initial[b].size()*sizeof(float));
            std::memset(ple_initial[b].data(),0xff,ple_initial[b].size()*sizeof(float));
            uint32_t seed = 0x731a29u + uint32_t(sequence)*719 + b*101;
            if (active[b]) {
                for (int gi = 0; gi < layers; ++gi) {
                    auto* base = initial[b].data()+guard+gi*state_stride;
                    for (int64_t i = 0; i < state_floats; ++i) base[i] = 0.002f*noise(seed);
                    for (int i = 0; i < 3*channels; ++i) base[conv_offset+i] = 0.05f*noise(seed);
                }
                for (int i = 0; i < hist_floats; ++i) ple_initial[b][guard+i] = noise(seed);
            }
            reference[b]->upload(initial[b]); candidate[b]->upload(initial[b]);
            ref_ple[b]->upload(ple_initial[b]); new_ple[b]->upload(ple_initial[b]);
        }
        const uint32_t seed = 0x135792u + uint32_t(sequence)*31;
        fill(weights,seed+1,0.12f); fill(gamma,seed+2,0.05f,1.0f);
        fill(qkv,seed+3,0.2f); fill(gate,seed+4,0.1f,-0.2f);
        fill(beta,seed+5,0.1f,0.5f); fill(z,seed+6,0.3f); fill(normalized,seed+7,0.7f);
        ref_h.poison(); new_h.poison(); ref_y.poison(); new_y.poison();
        ref_snap.poison(); new_snap.poison(); *status.host = 0;
        std::atomic_thread_fence(std::memory_order_seq_cst);
        // Reset uploads use the default stream; the graph streams are
        // nonblocking, so establish readiness explicitly before replay.
        ck(cudaDeviceSynchronize(), "fixture ready");
    }
    void legacy_target() {
        const auto& w = *staged_window.host;
        for (int gi = 0; gi < layers; ++gi) for (uint32_t i = 0; i < w.segments; ++i) {
            const auto& seg = w.segment[i]; const size_t f = seg.first;
            const size_t layer_row = size_t(gi)*rows+f;
            k::gdn_conv_l2_multi(state(seg.slot,gi)+conv_offset,qkv.p+layer_row*channels,
                weights.p+gi*channels*4,ref_h.p+layer_row*channels,channels,2*hk,1e-6f,seg.length,main);
            k::gdn_step_norm_multi(state(seg.slot,gi),ref_h.p+layer_row*channels,channels,
                gate.p+layer_row*hv,beta.p+layer_row*hv,z.p+layer_row*value_dim,
                gamma.p+gi*128,1e-6f,ref_y.p+layer_row*value_dim,hk,hv,seg.length,nullptr,main);
        }
        for (uint32_t row = 0; row < w.rows; ++row) {
            const int b = w.row[row].slot;
            k::ple_history_advance(history(b),normalized.p+row*k::NG_HC_DIM,main);
            ck(cudaMemcpyAsync(ref_snap.p+row*hist_floats,history(b),hist_floats*sizeof(float),
                               cudaMemcpyDeviceToDevice,main), "legacy snapshot");
        }
        ck(cudaStreamSynchronize(main), "legacy target sync");
    }
    void candidate_group(int begin, int end, cudaStream_t stream) {
        for (int gi = 0; gi < layers; ++gi) {
            const size_t f = size_t(gi)*rows;
            k::batch_device_gdn_conv(view,begin,end,gi,shape,qkv.p+f*channels,
                weights.p+gi*channels*4,new_h.p+f*channels,1e-6f,stream);
            k::batch_device_gdn_step(view,begin,end,gi,shape,new_h.p+f*channels,
                gate.p+f*hv,beta.p+f*hv,z.p+f*value_dim,gamma.p+gi*128,
                new_y.p+f*value_dim,1e-6f,stream);
        }
        for (int row = begin; row < end; ++row) {
            auto* snap = new_snap.p+row*hist_floats;
            k::batch_device_ple_copy(view,row,snap,hist_floats,false,stream);
            k::ple_history_advance(snap,normalized.p+row*k::NG_HC_DIM,stream);
            k::batch_device_ple_copy(view,row,snap,hist_floats,true,stream);
        }
    }
    static void finish_capture(Graph& graph, cudaStream_t stream) {
        ck(cudaStreamEndCapture(stream,&graph.graph), "end capture");
        ck(cudaGraphGetNodes(graph.graph,nullptr,&graph.nodes), "graph nodes");
        require(graph.nodes > 0,"empty candidate graph");
        ck(cudaGraphInstantiate(&graph.exec,graph.graph,nullptr,nullptr,0), "instantiate");
    }
    Graph& target_graph() {
        const int n = staged_window.host->rows, cut = staged_window.host->dense_cut;
        const auto key = std::make_pair(n,cut);
        auto it = targets.find(key);
        if (it != targets.end()) return *it->second;
        auto graph = std::make_unique<Graph>();
        ck(cudaStreamBeginCapture(main,cudaStreamCaptureModeThreadLocal), "target capture");
        k::copy_i32_from_mapped(reinterpret_cast<int32_t*>(window.p),
            reinterpret_cast<const int32_t*>(staged_window.device),sizeof(k::DeviceWindowLayout)/4,main);
        if (cut) {
            ck(cudaEventRecord(fork,main), "record fork");
            ck(cudaStreamWaitEvent(side,fork,0), "wait fork");
            candidate_group(0,cut,main); candidate_group(cut,n,side);
            ck(cudaEventRecord(join,side), "record join");
            ck(cudaStreamWaitEvent(main,join,0), "wait join");
        } else candidate_group(0,n,main);
        finish_capture(*graph,main);
        auto* ptr = graph.get(); targets.emplace(key,std::move(graph)); return *ptr;
    }
    Graph& get_commit_graph() {
        if (commit_graph) return *commit_graph;
        commit_graph = std::make_unique<Graph>();
        ck(cudaStreamBeginCapture(main,cudaStreamCaptureModeThreadLocal), "commit capture");
        k::copy_i32_from_mapped(reinterpret_cast<int32_t*>(commit.p),
            reinterpret_cast<const int32_t*>(staged_commit.device),sizeof(k::DeviceCommitLayout)/4,main);
        for (int gi = 0; gi < layers; ++gi) {
            const size_t f = size_t(gi)*rows;
            k::batch_device_gdn_commit(view,gi,shape,qkv.p+f*channels,new_h.p+f*channels,
                                      gate.p+f*hv,beta.p+f*hv,main);
        }
        k::batch_device_ple_commit(view,new_snap.p,hist_floats,main);
        finish_capture(*commit_graph,main); return *commit_graph;
    }
    void launch(Graph& graph, const char* label) {
        std::atomic_thread_fence(std::memory_order_seq_cst);
        ck(cudaGraphLaunch(graph.exec,main), label); ck(cudaStreamSynchronize(main), "graph sync");
        ++graph.launches;
    }
    void check_target() {
        require(*status.host == 0,"target descriptor status");
        equal(ref_h,new_h,"GDN conv+poison"); equal(ref_y,new_y,"GDN target+poison");
        equal(ref_snap,new_snap,"PLE post-advance snapshots+poison");
        const auto y = new_y.download();
        for (int gi = 0; gi < layers; ++gi)
            for (size_t i = size_t(gi)*rows*value_dim;
                 i < (size_t(gi)*rows+staged_window.host->rows)*value_dim; ++i)
                require(std::isfinite(y[i]),"nonfinite GDN target");
        for (int b = 0; b < 4; ++b) {
            equal(reference[b]->download(),initial[b],"legacy verify mutated GDN state");
            equal(candidate[b]->download(),initial[b],"table verify mutated GDN state");
            equal(*ref_ple[b],*new_ple[b],"PLE forward/inactive-owner guards");
        }
        check_ple_oracle(false);
    }
    void check_ple_oracle(bool committed) {
        const auto& w = *staged_window.host;
        const auto norm = normalized.download();
        const auto snapshots = new_snap.download();
        for (int b = 0; b < 4; ++b) {
            auto expected = ple_initial[b];
            for (uint32_t si = 0; si < w.segments; ++si) {
                const auto& seg = w.segment[si]; if (seg.slot != b) continue;
                const int count = committed ? staged_commit.host->keep_by_segment[si] : seg.length;
                for (int j = 0; j < count; ++j) {
                    const int row = seg.first+j;
                    for (int ch = 0; ch < k::NG_HC_DIM; ++ch) {
                        float* h = expected.data()+guard+ch*k::NG_HIST;
                        for (int t = 0; t < k::NG_HIST-1; ++t) h[t] = h[t+1];
                        h[k::NG_HIST-1] = norm[size_t(row)*k::NG_HC_DIM+ch];
                    }
                    equal(std::vector<float>(expected.begin()+guard,expected.begin()+guard+hist_floats),
                          std::vector<float>(snapshots.begin()+size_t(row)*hist_floats,
                                             snapshots.begin()+size_t(row+1)*hist_floats),
                          "independent PLE row-fastest snapshot oracle");
                }
            }
            equal(new_ple[b]->download(),expected,"independent PLE history/guard oracle");
        }
    }
    void reject_stale_commit() {
        std::array<std::vector<float>,4> before, ple_before;
        for (int b = 0; b < 4; ++b) {
            before[b] = candidate[b]->download(); ple_before[b] = new_ple[b]->download();
        }
        const auto valid = *staged_commit.host;
        ++staged_commit.host->sequence;
        launch(get_commit_graph(),"stale commit");
        require(*status.host == k::kBatchBadCommit,"stale commit must fail closed");
        for (int b = 0; b < 4; ++b) {
            equal(candidate[b]->download(),before[b],"stale commit wrote GDN");
            equal(new_ple[b]->download(),ple_before[b],"stale commit wrote PLE");
        }
        *staged_commit.host = valid; *status.host = 0; ++rejected_commits;
    }
    void legacy_commit() {
        const auto& w = *staged_window.host;
        for (int gi = 0; gi < layers; ++gi) for (uint32_t i = 0; i < w.segments; ++i) {
            const auto& seg = w.segment[i]; const size_t f = size_t(gi)*rows+seg.first;
            k::gdn_conv_commit(state(seg.slot,gi)+conv_offset,qkv.p+f*channels,channels,keeps.p+seg.slot,main);
            k::gdn_step_norm_multi(state(seg.slot,gi),ref_h.p+f*channels,channels,
                gate.p+f*hv,beta.p+f*hv,z.p+f*value_dim,gamma.p+gi*128,1e-6f,
                ref_y.p+f*value_dim,hk,hv,seg.length,keeps.p+seg.slot,main,seg.length);
        }
        for (uint32_t i = 0; i < w.segments; ++i) {
            const auto& seg = w.segment[i];
            const int row = seg.first+staged_commit.host->keep_by_segment[i]-1;
            ck(cudaMemcpyAsync(history(seg.slot),ref_snap.p+row*hist_floats,
                hist_floats*sizeof(float),cudaMemcpyDeviceToDevice,main),"legacy PLE commit");
        }
        ck(cudaStreamSynchronize(main),"legacy commit sync");
    }
    void run() {
        for (const auto& layout : layouts) for (int pattern = 0; pattern < 4; ++pattern) {
            std::snprintf(context,sizeof(context),"hk=%d hv=%d layout=%s prefix=%d",
                          hk,hv,layout.name,pattern);
            reset(layout,pattern); legacy_target();
            auto& graph = target_graph(); launch(graph,"candidate target"); ++target_launches;
            check_target();
            if (pattern == 0) reject_stale_commit();
            legacy_commit(); launch(get_commit_graph(),"candidate accepted-prefix commit");
            require(*status.host == 0,"accepted-prefix descriptor status");
            for (int b = 0; b < 4; ++b) {
                equal(*reference[b],*candidate[b],"GDN accepted-prefix state/history/layer guards");
                equal(*ref_ple[b],*new_ple[b],"PLE accepted-prefix/inactive-owner guards");
            }
            equal(ref_y,new_y,"commit must not overwrite target outputs");
            check_ple_oracle(true); ++cases;
        }
        require(targets.size() == 4,"expected only R12/cut6, R12/cut8, R6/cut0, R8/cut0");
        require(targets.at({12,6})->launches == 20,"R12/cut6 must reuse one graph across nseg/owners");
        require(targets.at({12,8})->launches == 20,"R12/cut8 must reuse one graph across nseg/owners");
        require(target_launches == layouts.size()*4,"target replay count");
        std::printf("geometry hk=%d hv=%d: %zu target graphs, one fixed-four-segment commit graph\n",hk,hv,targets.size());
        for (const auto& [key,g] : targets)
            std::printf("  R=%d cut=%d nodes=%zu launches=%zu\n",key.first,key.second,g->nodes,g->launches);
    }
};
} // namespace

int main() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices < 1) {
        std::fprintf(stderr,"SKIP: NVIDIA CUDA device unavailable\n"); return 77;
    }
    if (!k::batch_device_gdn_supported() || !k::ple_block_available()) {
        std::fprintf(stderr,"SKIP: requires supported production GDN profile and PLE CUDA path\n"); return 77;
    }
    size_t free = 0, total = 0;
    ck(cudaMemGetInfo(&free,&total),"memory info");
    if (free < (8ull<<30)+(256ull<<20)) {
        std::fprintf(stderr,"SKIP: preserving 8 GiB reserve with 256 MiB fixture allowance\n"); return 77;
    }
    { Fixture small(2,4); small.run(); }
    { Fixture production(16,48); production.run(); }
    std::printf("PASS: %zu GDN/PLE fixtures, %zu stale commits rejected, %zu bitwise region comparisons; "
                "actual 16/48-head GDN, 2 layers, 12-row stride, fork/join cached cross-owner/nseg replay\n",
                cases,rejected_commits,comparisons);
    std::printf("NOT COVERED: QSA Q4 append/Host/pool, indexer tail/dead/pooled, scores/IDs, "
                "union map, streamed 5120-slot attention, full-model sampling or performance\n");
}
