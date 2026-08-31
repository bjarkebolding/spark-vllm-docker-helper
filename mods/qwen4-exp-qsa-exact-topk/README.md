# qwen4-exp-qsa-exact-topk

Opt-in deterministic QSA block top-k. On sm_121 the QSA indexer falls back to
`torch.ops._C.persistent_topk`, which is non-deterministic run-to-run and can drop real
top-k candidates ([vllm#51782](https://github.com/vllm-project/vllm/issues/51782)). With
`VLLM_QSA_EXACT_TOPK=1` the indexer uses an exact `torch.topk` over the visible columns
instead.

Correctness, not speed — decode is unchanged, prefill is ~8% slower. The `w4a16` recipe
enables it. No-op unless `VLLM_QSA_EXACT_TOPK` is set.

Patcher (`patch_qsa_exact_topk.py`) by [blazux/qwen3.8-Flash-DGX](https://github.com/blazux/qwen3.8-Flash-DGX)
(Apache-2.0). Idempotent, aborts on an unrecognised `qsa.py`.
