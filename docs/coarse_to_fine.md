# Fixed coarse-to-fine frame selection

`observe_coarse_to_fine(reader, encode_images)` connects the decoder to the fixed
8+16 selection rule. The encoder callback sees only the eight coarse images,
in time order. It must return one embedding per image. There is no question-based
ranking and no second chance to choose an interval.

I use the actual decoded coarse timestamps as interval endpoints. The largest
adjacent cosine distance wins, with the earliest interval winning a tie. The
reader then decodes 16 interior midpoint requests, excluding all coarse frame
IDs. A collision or incomplete batch fails instead of padding the selection.
The returned 24 images are sorted by actual time for the eventual answer prompt.

The result retains coarse IDs and fine requests so the selection can be audited.
`encoder_latency_ms` measures the callback; a GPU callback must synchronize before
returning. `selection_latency_ms` includes that time plus both decoding batches
and selection checks. Do not add the two values together. Initial reader indexing,
encoder loading and VLM inference are outside this measurement.

The callback boundary is tested with synthetic embeddings. The frozen CLIP
ViT-B/32 adapter, its preprocessing/revisions, real inference and baseline answer
prompts are still pending. This is not a measured CLIP benchmark result.
