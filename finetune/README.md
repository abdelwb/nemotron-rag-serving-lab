# Fine-tuning

LoRA supervised fine-tune of [`nvidia/Nemotron-Mini-4B-Instruct`](https://huggingface.co/nvidia/Nemotron-Mini-4B-Instruct) on a subset of [`nvidia/Daring-Anteater`](https://huggingface.co/datasets/nvidia/Daring-Anteater), using `transformers` + `peft` + `trl` + `accelerate` + `bitsandbytes`.

Run in order, in Google Colab:

1. **[`01_prepare_dataset.ipynb`](01_prepare_dataset.ipynb)** — CPU runtime is fine. Downloads and formats the dataset, saves a train/eval split.
2. **[`02_lora_finetune.ipynb`](02_lora_finetune.ipynb)** — switch to a T4 GPU runtime. Trains the adapter and pushes it to your own HF namespace.

Both notebooks call `huggingface_hub.login()`, which needs a free account and an access token from [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) (read scope for notebook 1, write scope for notebook 2, since it pushes the trained adapter). If `Nemotron-Mini-4B-Instruct` is gated on the Hub, accept its terms on the model page first.

See [`../docs/architecture.md`](../docs/architecture.md) for the full methodology and expected runtimes.

## Local install (only if you have your own GPU)

```bash
pip install -r requirements.txt
```
