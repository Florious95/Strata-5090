#include "strata/kernels/batch_deterministic_reduction.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
#include <stdexcept>

namespace k = strata::kernels;
namespace {
int checks = 0;
void require(bool ok, const char* label) {
    ++checks;
    if (!ok) { std::fprintf(stderr, "FAIL check %d: %s\n", checks, label); std::exit(1); }
}
void environment(const char* value) {
#if defined(_WIN32)
    require(_putenv_s("STRATA_BATCH_DETERMINISTIC_REDUCTION", value ? value : "") == 0, "set environment");
#else
    require((value ? setenv("STRATA_BATCH_DETERMINISTIC_REDUCTION", value, 1)
                   : unsetenv("STRATA_BATCH_DETERMINISTIC_REDUCTION")) == 0, "set environment");
#endif
}
} // namespace

int main(int argc, char** argv) {
    if (argc != 2) return 2;
    for (bool initial : {false, true}) {
        bool result = initial;
        require(k::parse_batch_deterministic_reduction(nullptr, result) && !result, "absent defaults OFF");
        result = initial;
        require(k::parse_batch_deterministic_reduction("0", result) && !result, "explicit OFF");
        result = initial;
        require(k::parse_batch_deterministic_reduction("1", result) && result, "explicit ON");
        for (const char* bad : {"", "2", "-1", "+1", "01", "00", "true", "false", " 1", "1 ", "1\n", "0x1"}) {
            result = initial;
            require(!k::parse_batch_deterministic_reduction(bad, result) && result == initial,
                    "reject noncanonical values without mutating result");
        }
    }

    bool expected = false;
    if (!std::strcmp(argv[1], "unset")) environment(nullptr);
    else if (!std::strcmp(argv[1], "0")) environment("0");
    else if (!std::strcmp(argv[1], "1")) { environment("1"); expected = true; }
    else if (!std::strcmp(argv[1], "invalid")) {
        for (const char* bad : {"", "2", "01", "1 "}) {
#if defined(_WIN32)
            // Windows treats an empty environment assignment as removal.
            if (!*bad) continue;
#endif
            environment(bad);
            bool rejected = false;
            try { (void) k::batch_deterministic_reduction_enabled(); }
            catch (const std::invalid_argument& e) {
                rejected = std::strcmp(e.what(), "STRATA_BATCH_DETERMINISTIC_REDUCTION must be 0 or 1") == 0;
            }
            require(rejected, "invalid profile throws before numerical dispatch");
        }
        environment("1"); expected = true;
    } else return 2;

    require(k::batch_deterministic_reduction_enabled() == expected, "first query freezes requested/default profile");
    for (const char* later : {"0", "1", "invalid", ""}) {
        environment(later);
        require(k::batch_deterministic_reduction_enabled() == expected, "graph profile immutable after initialization");
    }
    environment(nullptr);
    require(k::batch_deterministic_reduction_enabled() == expected, "unsetting environment cannot change captured profile");
    std::printf("PASS batch_deterministic_reduction_cpu mode=%s frozen=%d: %d checks\n", argv[1], int(expected), checks);
}
