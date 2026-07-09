# NOTES

## NLA repo/checkpoint verification

Source repo: `github.com/kitft/natural_language_autoencoders`

Relevant docs:

- `README.md`
- `docs/inference.md`
- standalone reference client: `nla_inference.py`

Verified public Gemma-3-12B checkpoint pair:

- source model: `google/gemma-3-12b-it`
- extraction layer: 32 of 48
- d_model: 3840
- AV: `kitft/nla-gemma3-12b-L32-av`
- AR: `kitft/nla-gemma3-12b-L32-ar`

Runtime contract:

- Each checkpoint ships `nla_meta.yaml`.
- Load the sidecar at runtime; do not hardcode prompt templates, token ids, or scale factors.
- The AV receives exactly one activation vector injected as one token embedding.
- The sidecar AV prompt template must be used exactly.
- Tokenize the AV prompt with one-step `tokenizer.apply_chat_template(..., tokenize=True, add_generation_prompt=True)`.
- Do not call `apply_chat_template(tokenize=False)` followed by `encode(add_special_tokens=True)` for AV prompts, because Gemma/Llama templates already include BOS and positions can shift.
- Rescale the raw activation to sidecar `extraction.injection_scale` before injection.
- For Gemma-3-12B, the expected orientation values are injection char `㈜`, token id `246566`, and injection scale `80000`, but code asserts against the sidecar instead of trusting these constants.
- The injection position is found by scanning for the sidecar injection token id and verifying both neighbor token ids.
- AV outputs should contain `<explanation>...</explanation>`.
- Mostly CJK output is a likely injection-failure signal because the injection marker is a CJK character.
- Early sequence positions, roughly the first 10 tokens, are known noisy inputs and should not be over-interpreted.

AR scoring:

- The AR reconstructs explanation text back to an activation vector.
- Directional MSE after both vectors are normalized to `sqrt(d_model)` equals `2 * (1 - cosine_similarity)`.
- Cosine around 0.9 is a good decode; around 0.5 is mediocre; around 0 is orthogonal.

## Server paths

- Code root: `/home/eagle0914/medical_nla`
- uv environment: `/data1/heejae/uv/medical_nla`
- Artifact root: `/data1/heejae/medical_nla`
- Hugging Face cache: `/data1/heejae/hf_cache`

## Memory log

Pending server run.

Record `nvidia-smi` memory after:

1. Gemma-3-12B-IT load with `bfloat16`, `device_map="cuda"`
2. dummy forward with `output_hidden_states=True`
3. Gemma unload and `torch.cuda.empty_cache()`
4. NLA checkpoint load
5. NLA generation

## Known constraints

- Do not use SGLang or vLLM.
- Default to two-pass execution because Gemma-3-12B-IT plus NLA may not fit in 32GB VRAM simultaneously.
- Do not add quantization unless explicitly approved, because it can change activation values.
- If SGLang is revisited later, Gemma-3 needs `--attention-backend fa3`, the upstream Gemma multimodal `input_embeds` patch, and `--disable-radix-cache`.
