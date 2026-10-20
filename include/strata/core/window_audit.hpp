#pragma once

// Opt-in host-side native accounting. No CUDA calls, token IDs or prompt data.
// Wire contract: docs/WINDOW-AUDIT-V1.md. The default-disabled path writes nothing.
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <map>
#include <string>
#include <utility>
#include <vector>
#include <fcntl.h>
#if defined(_WIN32)
#include <io.h>
#include <process.h>
#include <sys/stat.h>
#else
#include <unistd.h>
#endif

namespace strata::core {

struct AuditRoute {
    int64_t routed = 0, hits = 0, cpu_entries = 0, offload_entries = 0, skipped = 0;
    bool available = true;
    template<class Dispatch> static AuditRoute read(const Dispatch& d) {
        return {d.experts, d.cache_hits, d.multi_entries, d.offload_entries, d.tail_skipped, true};
    }
    bool add(const AuditRoute& r) {
        auto sum = [&](int64_t& a, int64_t b) {
            if (a < 0 || b < 0 || b > INT64_MAX - a) { available = false; return; }
            a += b;
        };
        available = available && r.available;
        sum(routed, r.routed); sum(hits, r.hits); sum(cpu_entries, r.cpu_entries);
        sum(offload_entries, r.offload_entries); sum(skipped, r.skipped);
        return available;
    }
    AuditRoute since(const AuditRoute& before) const {
        AuditRoute d = *this;
        auto delta = [&](int64_t a, int64_t b) {
            if (a < 0 || b < 0 || a < b) { d.available = false; return int64_t{0}; }
            return a - b;
        };
        d.available = available && before.available;
        d.routed = delta(routed, before.routed); d.hits = delta(hits, before.hits);
        d.cpu_entries = delta(cpu_entries, before.cpu_entries);
        d.offload_entries = delta(offload_entries, before.offload_entries);
        d.skipped = delta(skipped, before.skipped);
        return d;
    }
    bool valid() const {
        if (!available || routed <= 0 || hits < 0 || cpu_entries < 0 || offload_entries < 0 || skipped < 0) return false;
        // Subtraction, not a potentially overflowing sum of untrusted counters.
        int64_t left = routed;
        for (const auto x : {hits, cpu_entries, offload_entries, skipped}) {
            if (x > left) return false;
            left -= x;
        }
        return left == 0;
    }
};

struct AuditSlot {
    int id = 0, rows = 0, useful = 0, keep = 0;
    uint64_t generation = 0;
    int64_t position = 0, emitted = 0, produced_before = 0;
    const char* finish = nullptr;
};
struct AuditDispatch {
    bool canonical = false, generalized = false, merged = false, dense_parallel = false;
    bool shared_after_ring = false, head_wide = false, graph_hit = false;
    int dense_groups = 1, head_type = -1, qsa_tiles = 0, qsa_rows = 0, qsa_scalar = 0;
    int kv_mode = 0, scalar_resolves = 0, union_resolves = 0;
    const char* execution_path = "legacy";
    const char* graph_key_kind = "legacy_layout";
    int graph_key_rows = 0, graph_owner_slot = -1;
    uint64_t graph_address_epoch = 0;
};
struct AuditWindow {
    std::vector<int> active, row_slots;
    std::vector<AuditSlot> slots;
    AuditRoute before, route;
    AuditDispatch dispatch;
};
struct AuditConfig {
    int slots = 0, max_rows = 0;
    int64_t max_context = 0, resident_requested = 0;
    std::vector<int64_t> effective_resident;
    bool generalized = false, canonical = false, segmented = false, slot_solo = false;
    int solo_graph_limit = 0;
    std::string engine_version, source_sha;
    int source_dirty = -1;
};

class WindowAudit {
public:
    ~WindowAudit() { close(); }
    WindowAudit() = default;
    WindowAudit(const WindowAudit&) = delete;
    WindowAudit& operator=(const WindowAudit&) = delete;
    bool enabled() const { return file_ != nullptr; }
    bool good() const { return error_.empty(); }
    const std::string& error() const { return error_; }
    uint64_t generation(int b) const { return enabled() ? generations_.at((size_t) b) : 0; }
    void track_copies() { copy_instrumented_ = true; }
    // Logical historical KV + pooled-index payload, once per successful operation;
    // excludes checkpoint metadata and stream-page DMA (not a PCIe byte counter).
    enum Copy { IntoSlot, FromSlot, Park, Restore };
    void copied(Copy kind, size_t bytes) {
        if (!enabled() || admission_slot_ < 0) return;
        if (bytes > (size_t) INT64_MAX) { copy_known_ = false; return; }
        auto& total = copy_bytes_[(size_t) kind];
        if ((int64_t) bytes > INT64_MAX - total) copy_known_ = false;
        else total += (int64_t) bytes;
    }
    bool abort_admission(bool canonical) {
        return admission_slot_ < 0 || end_admission(0, false, "admission-error", canonical, false);
    }

    bool open(const char* path, const AuditConfig& c) {
        if (!path || !*path) return true;
        if (file_ || c.slots < 2 || c.slots > 8 || c.max_rows < c.slots || c.max_rows > 16 ||
            c.max_context < 1 || c.effective_resident.size() != (size_t) c.slots)
            return fail("invalid audit configuration");
        max_rows_ = c.max_rows;
#if defined(_WIN32)
        const int fd = _open(path, _O_WRONLY | _O_CREAT | _O_EXCL | _O_BINARY, _S_IREAD | _S_IWRITE);
        if (fd >= 0) file_ = _fdopen(fd, "wb");
        if (fd >= 0 && !file_) _close(fd);
#else
        const int fd = ::open(path, O_WRONLY | O_CREAT | O_EXCL, 0600);
        if (fd >= 0) file_ = fdopen(fd, "wb");
        if (fd >= 0 && !file_) ::close(fd);
#endif
        if (!file_) return fail("cannot create exclusive audit file");
        std::setvbuf(file_, buffer_.data(), _IOFBF, buffer_.size());
        run_ = std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + "-" +
#if defined(_WIN32)
               std::to_string(_getpid());
#else
               std::to_string(getpid());
#endif
        generations_.resize((size_t) c.slots);
        states_.resize((size_t) c.slots, 0);
        bool streaming = false;
        for (auto resident : c.effective_resident) streaming = streaming || resident > 0;
        std::string s = prefix("header");
        s += ",\"slots\":" + num(c.slots) + ",\"max_context\":" + num(c.max_context) +
             ",\"kv_format\":\"q4_0\",\"kv_resident_requested\":" + num(c.resident_requested) +
             ",\"kv_resident_effective_by_slot\":" + integers(c.effective_resident) +
             ",\"build\":{\"engine_version\":" + quote(c.engine_version) + ",\"source_sha\":" +
             (c.source_sha.empty() ? "null" : quote(c.source_sha)) + ",\"source_dirty\":" +
             (c.source_dirty < 0 ? "null" : boolean(c.source_dirty != 0)) + "},\"capabilities\":{" +
             "\"generalized\":" + boolean(c.generalized) + ",\"canonical_admission\":" + boolean(c.canonical) +
             ",\"segmented_verify\":" + boolean(c.segmented) + ",\"streaming\":" + boolean(streaming) +
             ",\"slot_solo\":" + boolean(c.slot_solo) + ",\"row_table\":false" +
             ",\"slot_solo_target_graph_limit\":" + num(c.solo_graph_limit) +
             ",\"slot_solo_commit_graph_limit\":" + num(c.slot_solo ? 4 : 0) +
             ",\"max_rows\":" + num(c.max_rows) + ",\"max_dense_groups\":2,\"audit_all_batch_paths\":true}," +
             "\"provenance\":{\"counter_schema\":\"expert-dispatch-occurrences-v1\",\"route_sources\":{" +
             "\"routed\":\"ExpertDispatch.experts\",\"hits\":\"ExpertDispatch.cache_hits\"," +
             "\"cpu_entries\":\"ExpertDispatch.multi_entries\",\"offload_entries\":\"ExpertDispatch.offload_entries\"," +
             "\"skipped\":\"ExpertDispatch.tail_skipped\"}," +
             "\"snapshot_boundary\":\"post-reset-pre-target/post-pool-pre-draft\"," +
             "\"emitted_source\":\"native-token-protocol-write\"},\"audit_mode\":\"minimal\"}";
        return write(s, 0);
    }

    bool begin_admission(int b) {
        if (!enabled()) return good();
        if (b < 0 || b >= (int) generations_.size() || admission_slot_ >= 0 ||
            (generations_[(size_t) b] && states_[(size_t) b] != 4) || generations_[(size_t) b] == UINT64_MAX)
            return fail("audit admission owner is not drained");
        ++generations_[(size_t) b];
        if (generations_[(size_t) b] > 1 && !lifecycle(b, "reused", nullptr)) return false;
        if (!lifecycle(b, "claim", nullptr)) return false;
        admission_slot_ = b; admission_route_ = {}; admission_runs_ = 0;
        copy_bytes_ = {}; copy_known_ = copy_instrumented_;
        ++admission_id_;
        return good();
    }
    void admission_target(const AuditRoute& route) {
        if (!enabled() || admission_slot_ < 0) return;
        ++admission_runs_;
        if (!route.valid()) admission_route_.available = false;
        admission_route_.add(route);
    }
    bool end_admission(int64_t emitted, bool continues, const char* finish, bool canonical,
                       bool completed = true) {
        if (!enabled()) return good();
        if (admission_slot_ < 0 || emitted < 0) return fail("audit admission completion has no owner");
        const int b = admission_slot_;
        const bool applicable = admission_runs_ > 0;
        const bool valid = completed && (!applicable || admission_route_.valid());
        if (!valid) ++invalid_;
        const std::string scope = applicable ? "admission_target" : "admission_no_target";
        std::string s = prefix("admission") + ",\"admission_id\":" + num(admission_id_) +
            ",\"scope\":" + quote(scope) + ",\"slot_id\":" + num(b) +
            ",\"request_generation\":" + num(generation(b)) + ",\"has_target_run\":" + boolean(applicable) +
            ",\"completed\":" + boolean(completed) + ",\"valid\":" + boolean(valid) + ",\"invalid_reason\":" +
            (valid ? "null" : quote(completed ? "native-route-counter-invalid" : "admission-incomplete")) +
            route_fields(admission_route_, applicable) +
            ",\"emitted\":" + num(emitted) + ",\"canonical_admission\":" + boolean(canonical) +
            ",\"copy_counter_valid\":" + boolean(copy_known_) +
            ",\"main_to_slot_kv_bytes\":" + optional(copy_bytes_[IntoSlot], copy_known_) +
            ",\"slot_to_main_kv_bytes\":" + optional(copy_bytes_[FromSlot], copy_known_) +
            ",\"park_bytes\":" + optional(copy_bytes_[Park], copy_known_) +
            ",\"restore_bytes\":" + optional(copy_bytes_[Restore], copy_known_) + "}";
        totals(scope, admission_route_, applicable, emitted);
        add(slot_totals_[{b, generation(b)}][1], emitted);
        admission_slot_ = -1;
        if (!write(s, 2)) return false;
        return continues ? lifecycle(b, "admitted", nullptr) : finish_slot(b, finish);
    }
    bool window(const AuditWindow& w) {
        if (!enabled()) return good();
        std::vector<bool> active(generations_.size()), scheduled_flags(generations_.size());
        for (int b : w.active) {
            if (b < 0 || b >= (int) active.size() || active[(size_t) b] || states_[(size_t) b] != 2)
                return fail("audit window has an invalid active owner");
            active[(size_t) b] = true;
        }
        size_t row = 0;
        for (const auto& slot : w.slots) {
            const int b = slot.id;
            if (b < 0 || b >= (int) active.size() || !active[(size_t) b] || scheduled_flags[(size_t) b] ||
                slot.generation != generation(b) || slot.rows < 1 || slot.rows > 8 ||
                slot.useful < 1 || slot.useful > slot.rows || slot.keep < 1 || slot.keep > slot.useful ||
                slot.position < 0 || slot.emitted < 0 || slot.emitted > slot.keep)
                return fail("audit window has an invalid scheduled segment");
            scheduled_flags[(size_t) b] = true;
            for (int j = 0; j < slot.rows; ++j, ++row)
                if (row >= w.row_slots.size() || w.row_slots[row] != b)
                    return fail("audit row ownership differs from scheduled segments");
        }
        if (row < 1 || row != w.row_slots.size() || row > (size_t) max_rows_)
            return fail("audit window exceeds configured row capacity");
        const bool valid = w.route.valid();
        if (!valid) ++invalid_;
        std::string s = prefix("window") + ",\"window_id\":" + num(++window_id_) +
            ",\"scope\":\"target_decode\",\"completed\":true,\"valid\":" + boolean(valid) +
            ",\"invalid_reason\":" + (valid ? "null" : quote("native-route-counter-invalid")) +
            ",\"stage_id\":0,\"device_id\":0,\"active_slot_ids\":" + integers(w.active) +
            ",\"row_slot_ids\":" + integers(w.row_slots) + ",\"rows\":" + num(w.row_slots.size());
        std::vector<int> scheduled;
        int useful = 0; int64_t emitted = 0;
        std::string slots = "[";
        for (const auto& slot : w.slots) {
            if (!scheduled.empty()) slots += ',';
            scheduled.push_back(slot.id); useful += slot.useful;
            add(emitted, slot.emitted);
            slots += "{\"slot_id\":" + num(slot.id) + ",\"request_generation\":" + num(slot.generation) +
                ",\"position_start\":" + num(slot.position) + ",\"rows\":" + num(slot.rows) +
                ",\"useful\":" + num(slot.useful) + ",\"keep\":" + num(slot.keep) +
                ",\"emitted\":" + num(slot.emitted) + ",\"finish_reason\":" +
                (slot.finish ? quote(slot.finish) : "null") + "}";
            add(slot_totals_[{slot.id, slot.generation}][0], slot.emitted);
        }
        slots += ']';
        const auto& d = w.dispatch;
        const char* strategy = d.scalar_resolves ? (d.union_resolves ? "mixed" : "scalar_per_row") :
                               d.union_resolves ? "union_per_slot" : "none";
        s += ",\"scheduled_slot_ids\":" + integers(scheduled) + ",\"rows_useful\":" + num(useful) +
            ",\"rows_padding\":" + num((int) w.row_slots.size() - useful) + ",\"slots\":" + slots +
            route_fields(w.route, true) + ",\"emitted\":" + num(emitted) + ",\"dispatch\":{" +
            "\"canonical_admission\":" + boolean(d.canonical) + ",\"generalized\":" + boolean(d.generalized) +
            ",\"merged\":" + boolean(d.merged) + ",\"dense_groups\":" + num(d.dense_groups) +
            ",\"dense_parallel\":" + boolean(d.dense_parallel) + ",\"shared_after_ring\":" + boolean(d.shared_after_ring) +
            ",\"head_type\":" + num(d.head_type) + ",\"head_wide_active\":" + boolean(d.head_wide) +
            ",\"qsa_tile_calls\":" + num(d.qsa_tiles) + ",\"qsa_tile_rows\":" + num(d.qsa_rows) +
            ",\"qsa_scalar_rows\":" + num(d.qsa_scalar) + ",\"kv_mode\":" + num(d.kv_mode) +
            ",\"kv_resolve_strategy\":" + quote(strategy) + ",\"kv_resolve_calls\":{\"scalar_per_row\":" +
            num(d.scalar_resolves) + ",\"union_per_slot\":" + num(d.union_resolves) + "},\"graph_cache_hit\":" +
            boolean(d.graph_hit) + ",\"execution_path\":" + quote(d.execution_path) +
            ",\"graph_key_kind\":" + quote(d.graph_key_kind) + ",\"graph_key_rows\":" + num(d.graph_key_rows) +
            ",\"graph_owner_slot\":" + num(d.graph_owner_slot) + ",\"graph_address_epoch\":" + num(d.graph_address_epoch) +
            ",\"fallback_reason\":" + (d.generalized ? "null" : quote("reference-policy")) + "}}";
        totals("target_decode", w.route, true, emitted);
        return write(s, 1);
    }
    bool finish_slot(int b, const char* reason) {
        if (!enabled()) return good();
        return lifecycle(b, "done", reason) && lifecycle(b, "drained", reason);
    }
    bool close() {
        if (!file_) return good();
        if (good()) {
            if (admission_slot_ >= 0) ++invalid_;
            for (size_t b = 0; b < generations_.size(); ++b)
                if (generations_[b] && states_[b] != 4) ++invalid_;
            std::string s = prefix("footer") + ",\"records_total\":" + num(seq_ + 1) +
                ",\"kind_counts\":{\"header\":" + num(counts_[0]) + ",\"window\":" + num(counts_[1]) +
                ",\"admission\":" + num(counts_[2]) + ",\"emit\":" + num(counts_[3]) +
                ",\"lifecycle\":" + num(counts_[4]) + ",\"footer\":1},\"invalid_records\":" + num(invalid_) +
                ",\"dropped_records\":0,\"scope_totals\":[";
            bool comma = false;
            for (const auto& [scope, t] : scope_totals_) {
                if (comma) s += ','; comma = true;
                s += "{\"scope\":" + quote(scope) + ",\"records\":" + num(t.records) +
                    ",\"route_records\":" + num(t.routes) + route_fields(t.route, t.routes > 0, false) +
                    ",\"emitted\":" + num(t.emitted) + "}";
            }
            s += "],\"slot_generation_totals\":["; comma = false;
            for (const auto& [key, t] : slot_totals_) {
                if (comma) s += ','; comma = true;
                int64_t total = 0; for (auto n : t) add(total, n);
                s += "{\"slot_id\":" + num(key.first) + ",\"request_generation\":" + num(key.second) +
                    ",\"window_emitted\":" + num(t[0]) + ",\"admission_emitted\":" + num(t[1]) +
                    ",\"other_emitted\":" + num(t[2]) + ",\"emitted\":" + num(total) + "}";
            }
            write(s + "]}", 5);
        }
        if (std::fflush(file_) != 0) fail("audit flush failed");
        if (std::fclose(file_) != 0) fail("audit close failed");
        file_ = nullptr;
        return good();
    }

private:
    struct Total { int64_t records = 0, routes = 0, emitted = 0; AuditRoute route; };
    std::FILE* file_ = nullptr;
    std::array<char, 65536> buffer_{};
    std::string run_, error_;
    uint64_t seq_ = 0, window_id_ = 0, admission_id_ = 0, invalid_ = 0;
    std::array<uint64_t, 6> counts_{};
    std::vector<uint64_t> generations_;
    std::vector<int> states_;   // 1 claim, 2 admitted, 3 done, 4 drained
    std::map<std::string, Total> scope_totals_;
    std::map<std::pair<int, uint64_t>, std::array<int64_t, 3>> slot_totals_;
    int admission_slot_ = -1, admission_runs_ = 0, max_rows_ = 0;
    bool copy_instrumented_ = false, copy_known_ = false;
    std::array<int64_t, 4> copy_bytes_{};
    AuditRoute admission_route_;
    bool fail(const char* why) { if (error_.empty()) error_ = why; return false; }
    bool add(int64_t& a, int64_t b) {
        if (b < 0 || a < 0 || b > INT64_MAX - a) return fail("audit integer counter overflow");
        a += b; return true;
    }
    bool lifecycle(int b, const char* event, const char* reason) {
        int& state = states_.at((size_t) b);
        const std::string e(event);
        if (e == "claim" && (state == 0 || state == 4)) state = 1;
        else if (e == "admitted" && state == 1) state = 2;
        else if (e == "done" && (state == 1 || state == 2)) state = 3;
        else if (e == "drained" && state == 3) state = 4;
        else if (e != "reused" || state != 4) return fail("invalid audit slot lifecycle transition");
        return write(prefix("lifecycle") + ",\"slot_id\":" + num(b) + ",\"request_generation\":" +
                     num(generation(b)) + ",\"event\":" + quote(e) + ",\"reason\":" +
                     (reason ? quote(reason) : "null") + "}", 4);
    }
    void totals(const std::string& scope, const AuditRoute& route, bool applicable, int64_t emitted) {
        auto& t = scope_totals_[scope];
        add(t.records, 1); add(t.emitted, emitted);
        if (applicable) {
            add(t.routes, 1);
            if (!route.valid()) t.route.available = false;
            t.route.add(route);
        }
    }
    bool write(const std::string& s, size_t kind) {
        if (!good() || seq_ == UINT64_MAX || counts_[kind] == UINT64_MAX) return fail("audit write/sequence failure");
        if (std::fwrite(s.data(), 1, s.size(), file_) != s.size() || std::fputc('\n', file_) == EOF)
            return fail("audit write failed");
        ++seq_; ++counts_[kind]; return true;
    }
    std::string prefix(const char* kind) const {
        return "{\"schema\":\"strata.window-audit.v1\",\"run_id\":" + quote(run_) +
               ",\"kind\":" + quote(kind) + ",\"seq\":" + num(seq_);
    }
    template<class T> static std::string num(T x) { return std::to_string(x); }
    static std::string boolean(bool x) { return x ? "true" : "false"; }
    static std::string optional(int64_t x, bool known) { return known ? num(x) : "null"; }
    template<class T> static std::string integers(const std::vector<T>& v) {
        std::string s = "[";
        for (size_t i = 0; i < v.size(); ++i) { if (i) s += ','; s += num(v[i]); }
        return s + ']';
    }
    static std::string quote(const std::string& value) {
        std::string s = "\"";
        const char* hex = "0123456789abcdef";
        for (const unsigned char c : value) {
            if (c == '"' || c == '\\') { s += '\\'; s += (char) c; }
            else if (c < 32) { s += "\\u00"; s += hex[c >> 4]; s += hex[c & 15]; }
            else s += (char) c;
        }
        return s + '"';
    }
    static std::string route_fields(const AuditRoute& r, bool applicable, bool include_scope = true) {
        const bool valid = applicable && r.valid();
        return ",\"route_applicable\":" + boolean(applicable) + ",\"route_counter_valid\":" +
            (applicable ? boolean(valid) : "null") +
            (include_scope ? ",\"route_scope\":" + std::string(applicable ? "\"target_computed_rows\"" : "null") : "") +
            ",\"routed\":" + optional(r.routed, valid) + ",\"hits\":" + optional(r.hits, valid) +
            ",\"cpu_entries\":" + optional(r.cpu_entries, valid) +
            ",\"offload_entries\":" + optional(r.offload_entries, valid) + ",\"skipped\":" + optional(r.skipped, valid);
    }
};

} // namespace strata::core
