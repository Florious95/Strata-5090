# Raw-image SHA256 references and idle-slot prefix affinity

Additive Python-only release based on source77cd2b2. Native CUDA/math/KV/checkpoint/GDN/M-RoPE contracts and the640c engine binary are unchanged.

## Legacy image inputs

Existing dataURL/base64, HTTP(S) and local-file inputs keep their parsing, normalization, token counts and error/security rules. SHA256 is computed from **raw decoded/loaded bytes, before normalization**. Raw→normalized and normalized→raw indexes share the existing SVE cache; no duplicate embeddings. A warm raw input skips normalization/ENC. Tool-image precheck results are reused during prepare, avoiding the previous second load/normalize. Failed legacy tool images still become an omitted-image note; failed user images still error.

## Opt-in references

After a successful original image request, a capable client can send:

```json
{"type":"image_url","image_url":{"url":"strata-sha256://<64 raw-byte SHA256 hex digits>"}}
```

`sha256:<64hex>` is also accepted; digest case is insensitive. Reference hits perform **no base64 decode, normalize, network/file input load or ENC**. They return the same SVE file/token count, preserving image order, native embedding identity and model input IDs. Request leases protect SVE files during tokenization/packing; hits bypass the unrelated legacy encoder FIFO. Invalid digest is400; a missing/evicted reference is409:

```json
{"error":{"type":"image_reference_error","code":"image_cache_miss",
 "message":"image reference is not cached; resend the original image","missing_sha256":["..."]}}
```

All references are preflighted before the legacy bad-tool-image fallback. A missing historical tool reference is **never silently replaced with text**. Retry once with only missing references replaced by their original images; do not blindly repeat inference or loop.

`GET /props` advertises `image_references`. Authenticated `GET /v1/vision/cache?sha256=HASH&sha256=HASH` supports up to64 digests; without hashes it reports capability/namespace. No encoder/model loading occurs for a query. Cache is volatile, per Vision instance, with64 SVE entries and64 raw aliases (leases may temporarily keep more SVE files alive); restart/eviction can require re-sending originals. Many differently encoded raw images that normalize identically share one SVE but consume separate bounded aliases. Cache is shared within the existing service auth scope; this does **not** implement multi-tenant key isolation or expose an unauthenticated public image store. Existing auth/CORS remain; short references never authorize arbitrary local paths.

This server upgrade alone does not change Pi session persistence or make an old client stop sending base64. Pi must store attachments outside its huge conversation JSONL and send the negotiated small references; retain originals for cache misses. No history pruning/semantic loss is performed here.

## Slot routing

Production `StrataEngine.pick_slot` already stored `slot_held`, but only recognized an entire held state as an exact shorter prefix. Re-rendered thinking/tool replies or a shorter replay made that score zero and fell back to empty/LRU, allowing a session to ping-pong slots.

Now choose the **longest token LCP among idle slots**, capped at prompt length−1 because the last prompt token starts verify. Prefer an intact canonical shorter prefix when scores tie; otherwise preserve slot order. Busy slots are never reserved/waited for; no-common-prefix fallback, group512-token threshold and pipelined-hole policy stay unchanged. Native still validates token/image identity and chooses actual GDN checkpoint resume. LCP routing cannot guarantee reuse after true middle-history or image changes, and does not bypass checkpoint correctness.

CPU tests reproduce both actual shapes:49252 prompt vs slot1's50266 state and slot0'sold23575;73276 prompt vs slot0's73238 and slot1'sold50266. Old selector fails these scenes; new selector keeps the more-affine slot. The old whole-prompt routing assertion is updated to the deliberately requested LCP behavior, not removed. Fresh tokens include genuinely new input, so don't label all31824/32737 fresh tokens as lost old cache.

## Validation/deployment boundary

`serve.test_vision_refs` uses fake encoder SVE fixtures and real Pillow PNG/WebP normalization, no model/GPU. Legacy↔ref/mixed IDs and combined SVE bytes are exact. Tests cover raw/reverse alias eviction, stale files,64+ images, concurrent leases, errors/cancel cleanup, bad-tool compatibility, API401/409/200, CORS file refusal and zero encoder FIFO work for hits. `serve.test_slot_affinity` covers the two long-context red scenes, shorter/rewritten/exact/tie/busy/group/LRU behavior. CPU fixtures are not production vision quality or throughput acceptance.

Publish a new source/server/ops pin as one controlled package; do not edit the running file or mix pins. No CUDA rebuild is needed; reusing the unchanged640c binary is explicit. Bench owns restart, known-image old/ref/mixed output checks, missing-ref fallback, native image/prefix and real Pi timing. Native prefix reuse/performance and any Pi55-second or1.5GB session improvement remain actual-client acceptance, not consequences claimed by unit tests.
