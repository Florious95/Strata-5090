# C2 tail-skip occurrence accounting

One host-accounting fix, based on runtime77cd2b2/PR#3. No CUDA arithmetic, routing, skip policy, accepted tokens, Dense/Narrow/QSA, structures/ABI or production config changes.

The source counts requested token-expert occurrences before dispatch. `expert_source.cpp` kind-2 increments `tail_skipped` and contributes zero: it is neither GPU cache hit, CPU execution nor offload. The old slot-only check omitted that category and stopped an otherwise authorizedTAIL_SKIP7 experiment.

This patch captures `tail_skipped` at the same pre-verify boundary as hits/routed/cpu/offload, and uses only the completed target window delta:

`hits + cpu_entries + offload_entries + skipped == routed`

The positive routed/nonnegative-category/exact-equality guard remains fail-closed. Missing work is never inferred as skipped. A subtraction-based pure check also rejects overflow-sized inconsistent inputs without signed addition UB. Native `batch-round` and `batch-depth` add explicit `skipped`; no unit mixing with deduplicated misses/jobs.

Withtail0,skipped0 and the old accounting contract are unchanged. Adaptive acceptance still uses the original requested-routed denominator; skipped is **not** counted as hit. Any executed-only cache ratio must be separately named. Nonzero skipped is a **quality-tradeoff** receipt, not lossless long-context qualification.

CPU red/green: the original three-category formula failed the new positive-skip case (exit1). Corrected production helper passes1296 category combinations, missing/negative/overflow controls,9840 existing layouts/1078 C2 allocations/220 depth cases; Release andASan/UBSan bothPASS. CI adds the exact helper tests. These are not GPU/quality/performance results.

Release target handoff uses an independent candidate directory and nice10 CPU compile/link, retaining matching77cd static CUDA/core libraries only because no existing ABI/kernel definitions change. Precise source, commands, library digests and newbinary SHA must be captured; immutable77cd source/build/binary and production pins are not overwritten. Bench executes the old failure workload and checks explicit skipped balances. Python observer must also add the skipped category and reject losses rather than suppress the error.
