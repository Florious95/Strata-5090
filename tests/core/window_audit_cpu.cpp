// Native writer/accounting contracts, without a model, server or CUDA runtime.
#include "strata/core/window_audit.hpp"
#include "strata/core/batch_segments.hpp"
#include "strata/kernels/kv_stream.hpp"
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
using namespace strata::core;
int checks = 0;
void require(bool yes, const char* label) {
    ++checks;
    if (!yes) { std::fprintf(stderr, "FAIL %s\n", label); std::exit(1); }
}
int main(int argc, char** argv) {
    require(argc <= 2, "optional owned output path");
    const std::string path = argc == 2 ? argv[1] :
        (std::filesystem::temp_directory_path() / ("strata-window-audit-cpu-" +
         std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + ".jsonl")).string();
    AuditRoute a{1920, 1824, 96, 0, 0};
    require(a.valid(), "independent native route conservation");
    AuditRoute invalid{1920, 1824, 97, 0, 0};
    require(!invalid.valid(), "nonconservation rejected");
    require(!AuditRoute{INT64_MAX, INT64_MAX, INT64_MAX, 0, 0}.valid(), "sum overflow cannot pass");
    require(!AuditRoute{1, -1, 2, 0, 0}.valid(), "negative counts rejected");
    require(!a.since({1921, 0, 0, 0, 0}).valid(), "cross-reset subtraction rejected");
    AuditRoute sum{INT64_MAX, INT64_MAX, 0, 0, 0};
    require(!sum.add({1, 1, 0, 0, 0}), "aggregation overflow rejected");
    int widths[3] = {6, 6, 4};
    const int packed = pack_segment_rows(widths, 3, 0);
    require(packed == 14 && widths[0] == 6 && widths[1] + widths[2] == 8, "packability backoff");
    int c4[4] = {3, 3, 3, 3}, c8[8] = {2, 2, 2, 2, 2, 2, 2, 2};
    require(pack_segment_rows(c4, 4, 0) == 12, "C4 twelve rows");
    require(pack_segment_rows(c8, 8, 0) == 16, "C8 sixteen rows");
    int negative[2] = {-1, 2};
    require(pack_segment_rows(negative, 2, 0) < 0, "negative width rejected");
    require(strata::kernels::kv_stream_union_fits(5120, 8, 2051, 4), "eight queries fit minimum streaming pool");
    require(!strata::kernels::kv_stream_union_fits(5120, 16, 2051, 4), "illegal full-window union rejected");
    require(!strata::kernels::kv_stream_union_fits(INT64_MAX, 1, INT64_MAX, 1), "union bound does not overflow");
    // Existing allocator outputs are bounded by 16; exhaust holes and asymmetric
    // widths inside that envelope rather than raising any kernel's capacity.
    for (int n : {2, 4, 8}) {
        const int radix = n == 8 ? 3 : 7;
        int cases = 1; for (int i = 0; i < n; ++i) cases *= radix;
        for (int code = 0; code < cases; ++code) {
            int original[8] = {}, w[8] = {}, value = code, total = 0;
            for (int i = 0; i < n; ++i) { original[i] = w[i] = value % radix; value /= radix; total += w[i]; }
            if (total > 16) continue;
            const int result = pack_segment_rows(w, n, code % n);
            require(result >= 0 && result <= total, "bounded packed row count");
            int ids[16] = {}, used = 0;
            for (int i = 0; i < n; ++i) {
                require(w[i] <= original[i] && (original[i] ? w[i] >= 1 : w[i] == 0), "pending rows survive backoff");
                for (int j = 0; j < w[i]; ++j) ids[used++] = i;
            }
            require(used == result && segment_group_boundary(ids, used) >= 0, "every emitted layout is packable");
        }
    }
    AuditConfig c;
    c.slots = 2; c.max_rows = 12; c.max_context = 262144; c.resident_requested = 20480;
    c.effective_resident = {20480, 20480}; c.engine_version = "synthetic-cpu-test";
    WindowAudit writer;
    require(writer.open(path.c_str(), c), "create exclusive private file");
    writer.track_copies();
    WindowAudit collision;
    require(!collision.open(path.c_str(), c), "existing file never overwritten");
    require(writer.begin_admission(0), "claim slot zero");
    writer.admission_target({480, 456, 24, 0, 0});
    writer.copied(WindowAudit::IntoSlot, 1024);
    writer.copied(WindowAudit::IntoSlot, 2048);
    writer.copied(WindowAudit::Park, 4096);
    writer.copied(WindowAudit::Restore, 8192);
    require(writer.end_admission(1, true, nullptr, false), "admission target output");
    require(writer.begin_admission(1), "claim slot one");
    require(writer.end_admission(0, true, nullptr, false), "no-target admission remains null route");
    AuditWindow w;
    w.active = {0, 1}; w.row_slots = {0, 0, 1, 1}; w.route = a;
    AuditSlot zero; zero.id = 0; zero.rows = zero.useful = zero.keep = 2;
    zero.generation = 1; zero.position = 258000; zero.emitted = 2; zero.finish = "length";
    AuditSlot one = zero; one.id = 1; one.keep = 1; one.emitted = 1;
    w.slots = {zero, one};
    w.dispatch.graph_key_rows = 4;
    w.dispatch.head_type = 13; w.dispatch.qsa_scalar = 48; w.dispatch.kv_mode = 1;
    w.dispatch.scalar_resolves = 24; w.dispatch.union_resolves = 12;
    require(writer.window(w), "write native window");
    require(writer.finish_slot(0, "length"), "slot zero drains");
    require(writer.finish_slot(1, "length"), "slot one drains");
    require(writer.begin_admission(0) && writer.generation(0) == 2, "reuse increments native generation");
    require(writer.end_admission(0, false, "cancel", false), "pre-admission cancel can drain");
    require(writer.close(), "flush footer");
    {
        std::ifstream in(path);
        const std::string contents((std::istreambuf_iterator<char>(in)), {});
        require(contents.find("\"execution_path\":\"legacy\",\"graph_key_kind\":\"legacy_layout\",\"graph_key_rows\":4,\"graph_owner_slot\":-1,\"graph_address_epoch\":0") != std::string::npos,
                "legacy graph identity is explicit, not guessed from rows");
    }
    {
        const std::string solo_path = path + ".solo";
        AuditConfig four = c;
        four.slots = 4; four.effective_resident = {20480, 20480, 20480, 20480};
        four.slot_solo = true; four.solo_graph_limit = 4;
        WindowAudit solo;
        require(solo.open(solo_path.c_str(), four) && solo.begin_admission(2), "solo arbitrary native owner claims");
        require(solo.end_admission(0, true, nullptr, true), "solo admission preserves native generation");
        AuditWindow one_slot;
        one_slot.active = {2}; one_slot.row_slots = {2, 2, 2};
        AuditSlot slot = zero; slot.id = 2; slot.rows = slot.useful = 3; slot.keep = slot.emitted = 2;
        one_slot.slots = {slot}; one_slot.route = {1440, 1368, 72, 0, 0};
        auto& d = one_slot.dispatch;
        d.canonical = d.generalized = true; d.execution_path = d.graph_key_kind = "slot_solo";
        d.graph_key_rows = 3; d.graph_owner_slot = 2; d.graph_address_epoch = 1;
        require(solo.window(one_slot), "solo window uses ordinary integer and generation accounting");
        require(solo.finish_slot(2, "length") && solo.close(), "solo lifecycle/footer closes");
        std::ifstream in(solo_path);
        const std::string contents((std::istreambuf_iterator<char>(in)), {});
        require(contents.find("\"slot_solo\":true,\"row_table\":false,\"slot_solo_target_graph_limit\":4,\"slot_solo_commit_graph_limit\":4") != std::string::npos,
                "header exposes experimental graph capability and count bounds");
        require(contents.find("\"execution_path\":\"slot_solo\",\"graph_key_kind\":\"slot_solo\",\"graph_key_rows\":3,\"graph_owner_slot\":2,\"graph_address_epoch\":1") != std::string::npos,
                "solo graph identity serializes exact owner/rows/epoch");
        require(std::filesystem::remove(solo_path), "remove owned solo artifact");
    }
    for (int mode = 0; mode < 4; ++mode) {
        const std::string bad_path = path + ".negative-" + std::to_string(mode);
        WindowAudit bad;
        require(bad.open(bad_path.c_str(), c) && bad.begin_admission(0), "negative fixture owns a private file");
        if (mode == 0) {
            require(bad.abort_admission(false), "failed admission drains instead of stranding its owner");
            require(bad.begin_admission(0) && bad.generation(0) == 2, "failed request can be followed by a new native generation");
            require(bad.end_admission(0, false, "cancel", false), "replacement pre-target request drains");
            require(bad.close(), "invalid admission is counted in a readable footer");
        } else {
            require(bad.end_admission(0, true, nullptr, false), "negative fixture admitted");
            AuditWindow invalid_window = w;
            invalid_window.active = {0}; invalid_window.slots = {zero}; invalid_window.row_slots = {0, 0};
            if (mode == 1) invalid_window.slots[0].generation = 2;
            if (mode == 2) invalid_window.row_slots[1] = 1;
            if (mode == 3) invalid_window.slots[0].emitted = 3;
            require(!bad.window(invalid_window), "stale generation, foreign rows and impossible emits fail closed");
            require(!bad.close(), "malformed campaign cannot have a successful close");
        }
        require(std::filesystem::remove(bad_path), "remove owned negative fixture");
    }
    if (argc == 1) require(std::filesystem::remove(path), "remove only this test's generated artifact");
    std::printf("PASS %d CPU accounting/writer/row/union checks; JSONL=%s\n", checks, path.c_str());
}
