# qwen4-exp-int8-lmhead

Passes `quant_config` to `ParallelLMHead` in `qwen4_exp/nvidia/model.py` **and**
`mtp.py`, so a checkpoint whose `lm_head` is quantized (int8 GPTQ in the `w4a16`
checkpoint) doesn't get forced back to a 1.27 GiB bf16 head + a bf16 GEMV/token.
Without the `mtp.py` half, `num_speculative_tokens >= 3` crashes at load
("no parameter named 'lm_head.qweight'"). Idempotent, shape-guarded, no-op on an
unquantized head.
