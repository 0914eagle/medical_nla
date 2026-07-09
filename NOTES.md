# NOTES

## OpenNLA repo/checkpoint verification

Pending. The task brief did not include the public OpenNLA Hugging Face repo id or GitHub URL.

Before running Phase 1/2 on the server, inspect the official OpenNLA repo and record:

- activation extraction location: target layer, residual stream location, and whether "layer L" maps to `hidden_states[L]` or `hidden_states[L + 1]`
- activation injection format: placeholder token text/id, number of placeholder tokens, and whether one activation vector or a span is expected
- NLA query template
- checkpoint format: full model vs. base model plus LoRA adapter
- required activation normalization/projection, if any

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
