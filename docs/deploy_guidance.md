# Kimi-K2.5 Deployment Guide

> [!Note]
> This guide only provides some examples of deployment commands for Kimi-K2.5, which may not be the optimal configuration. Since inference engines are still being updated frequenty,  please continue to follow the guidance from their homepage if you want to achieve better inference performance.

> kimi_k2 reasoning parser and other related features have been merged into vLLM/sglang and will be available in the next release. For now, please use the nightly build Docker image.
## vLLM Deployment

This model is available in nightly vLLM wheel:
```
uv pip install -U vllm \
    --torch-backend=auto \
    --extra-index-url https://wheels.vllm.ai/nightly
```

Here is the example to serve this model on a H200 single node with TP8 via vLLM:
```bash
vllm serve $MODEL_PATH -tp 8 --mm-encoder-tp-mode data --trust-remote-code --tool-call-parser kimi_k2 --reasoning-parser kimi_k2
```

### Long-Context (256K) Serving with FP8 KV Cache

To serve Kimi-K2.5 at its full 256K context length (`262,144` tokens) or when using deep Thinking Mode without running out of GPU memory (CUDA OOM), you can use the structured vLLM recipe with FP8 KV-cache enabled:

```bash
vllm serve moonshotai/Kimi-K2.5 --config docs/vllm_k2.5_fp8_kv.yaml

**Key notes**
- `--tool-call-parser kimi_k2`: Required for enabling tool calling
- `--reasoning-parser kimi_k2`: Kimi-K2.5 enables thinking mode by default. Make sure to pass this for correct reasoning processing.

## SGLang Deployment

This model is available in SGLang latest main:

```
pip install "sglang @ git+https://github.com/sgl-project/sglang.git#subdirectory=python"
pip install nvidia-cudnn-cu12==9.16.0.29
```

Similarly, here is the example for it to run with TP8 on H200 in a single node via SGLang:
``` bash
sglang serve --model-path $MODEL_PATH --tp 8 --trust-remote-code --tool-call-parser kimi_k2 --reasoning-parser kimi_k2
```
**Key parameter notes:**
- `--tool-call-parser kimi_k2`: Required when enabling tool usage.
- `--reasoning-parser kimi_k2`: Required for correctly processing reasoning content.

## KTransformers Deployment
### KTransformers+SGLang Inference Deployment
Launch with KTransformers + SGLang for CPU+GPU heterogeneous inference:

```
python -m sglang.launch_server \
  --model path/to/Kimi-K2.5/ \
  --kt-amx-weight-path path/to/Kimi-K2.5/ \
  --kt-cpuinfer 64 \
  --kt-threadpool-count 2 \
  --kt-num-gpu-experts 180 \
  --kt-amx-method AMXINT4 \
  --trust-remote-code \
  --mem-fraction-static 0.98 \
  --chunked-prefill-size 16384 \
  --max-running-requests 48 \
  --max-total-tokens 50000 \
  --tensor-parallel-size 8 \
  --enable-p2p-check \
  --disable-shared-experts-fusion
```

Achieves 640.12 tokens/s Prefill and 24.51 tokens/s Decode (48-way concurrency) on 8× NVIDIA L20 + 2× Intel 6454S.

More details: https://github.com/kvcache-ai/ktransformers/blob/main/doc/en/Kimi-K2.5.md .

### KTransformers+LLaMA-Factory Fine-tuning Deployment

You can use below command to run LoRA SFT with KT+llamafactory.

```
# For LoRA SFT
USE_KT=1 llamafactory-cli train examples/train_lora/kimik2_lora_sft_kt.yaml
# For Chat with model after LoRA SFT
llamafactory-cli chat examples/inference/kimik2_lora_sft_kt.yaml
# For API with model after LoRA SFT
llamafactory-cli api examples/inference/kimik2_lora_sft_kt.yaml
```

This achieves end-to-end LoRA SFT Throughput: 44.55 token/s on 2× NVIDIA 4090 + Intel 8488C with 1.97T RAM and 200G swap memory.

More details refer to https://github.com/kvcache-ai/ktransformers/blob/main/doc/en/SFT_Installation_Guide_KimiK2.5.md .
