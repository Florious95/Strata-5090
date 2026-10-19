# Images in canonical C2 slots

This candidate extends the qualified c15c25f v0.1.43 golden **text** stack. It is not a claim that image/GPU/production acceptance already passed.

## Contract

- `--batch-slot-only --vision` can use the wide C2 MTP fast path. Other legacy vision batching keeps its previous gate.
- BGENI binds the requested canonical target and its private drafter before multimodal prefill and the first token. There is no third solo session/KV or slot copy.
- Each of the two target sessions holds a non-owning reference to a stable independent image M-RoPE bank and its host counterpart. At262144 context each bank is about3MiB. Admission updates only the requested bank; bank addresses do not change after graph capture.
- Generated text has per-slot rotary position `cell + rope_delta`. Text chunks inside the prompt read that prompt's exact position from its bank. KV indices, causal steps, append order and accepted/rollback cells remain unshifted.
- Merged/Narrow query/key rotation captures **no global M-RoPE table**, using explicit per-row positions. The indexer pooling/commit captures the correct slot bank, including blocks straddling image/text. MTP image-prefill captures its own bank; MTP catch-up/decode captures no table and uses its parent slot's explicit positions.
- Host `MropeTableScope` only selects launch/capture arguments and restores the previous pointer. It does not change pointers in already captured kernels. No GPU hot-path allocation, new reduction or expert arithmetic is introduced.
- Session checkpoint/RAM snapshot save and restore the rotary delta; image identities remain attached to each slot. Existing image hot-slot reuse remains conservatively disabled rather than matching pictures by token ids alone.
- The current disk format cannot encode the new C2 rotary offset. SAVE of nonzero-offset images refuses **before filesystem access** instead of silently dropping it. Ordinary zero-offset/text file bytes retain their existing golden hash. Disk-format extension is outside this patch.

Dense/Narrow two-stream dispatch, Q5/Q6 head arithmetic, QSA tile, GDN, Direct Slot session count, spec6/rows12 and NaN guard remain unchanged.

## Production treatment

Use the already prepared **CPU** `strata-vision`, matching mmproj and model-vocabulary metadata. The vision object must contain executable/mmproj/model paths and `gpu: false`; preserve the tested token cap and explicitly choose encoder threads with the operator. No unknown `vision_on_demand` key creates an on-demand lifecycle. The encoder is a resident CPU child, not another full GPU model.

Keep reserve1400 initially: the new two image banks cost about6MiB total; CPU encoding must not create a CUDA context. Actual first/max/multi-image VRAM peaks and CPU contention still require measurement. If margins fail, change configuration explicitly and requalify—do not claim the old209.99 text measurement applies unchanged.

Build the candidate separately while the live binary/source pins remain untouched. Stop the owned production control group only in the benchmark owner's cutover window, preserve the old source/binary/config SHA for rollback, publish the new source and Release atomically, update launcher SHA pins and production JSON integrity binding, then restart. Matching server source is unchanged. Never print/copy API keys or launch a competing model.

Native observability: INFO `slot_vision=1`; admission stderr `strata batch vision: slot=N images=M rope_delta=D direct=1 solo_kv=0`. An image HTTP200 alone is not proof of two GPU-active slots or semantic identification.

## Tests and acceptance

- `slot_mrope_cpu`: production host-position helper, distinct real image-grid deltas, layouts1/7/8/9/10/11/12, near262K positions and nested/disabled capture-pointer scopes; Release and ASan/UBSan.
- `slot_mrope_gpu`: hardware bit comparison of per-slot bank oracle against explicit merged/narrow positions, head dimensions128/256, native plain/fused norm-RoPE, none/linear/YaRN scaling,32/32K/258K positions, image+image/image+text/text+text. Independent-stream graph replay changes the global pointer to verify captured arguments stay isolated. No model/GGUF.
- Extended `conversation_snapshot_test`: RAM rotary-offset round trip and invalid offsets rejected before writes. `conversation_file_test`: old format golden bytes unchanged and offset SAVE leaves no file.
- Extended CPU server fake-protocol test: direct BGENI to slot0 and slot1, mixed image/text queue, no solo GEN/GENI and no leaked admission claim. These are protocol mocks, **not native model evidence**.
- Re-run affected existing CPU/server and short golden GPU contracts under approved resource windows.
- Real owner acceptance must identify a known image, exercise both slots with different image grids and mixed text/image, verify correct subsequent text/slot reuse/cancel, preserve direct/no-solo native receipts, and check auth/Tailscale, NaN/OOM/Xid and minfree. Full text258K regression/throughput remains separately bench-owned; synthetic bit contracts do not guarantee full model correctness or speed.

Status at initial commit: local position helper635 checks twice (Release/ASan/UBSan) and conversation_file202 checks passed;363 service tests passed after installing the missing pinned regex dependency into the isolated existing test venv. First full-suite attempt retained three regex import errors. Linux Release, hardware contracts and production images are pending; do not merge/deploy based only on this paragraph.
