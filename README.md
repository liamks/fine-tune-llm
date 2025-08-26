## Data
Data format: JSONL with a text field. Put files under data/.

## One-off run:
    export WANDB_API_KEY=...
    export HF_TOKEN=...
    python scripts/train.py --config configs/train.default.yaml

## Sweep:
    wandb login
    wandb sweep sweeps/sft_lr_batch_lora.yaml
    wandb agent <entity>/<project>/<SWEEP_ID>  # launch 4 of these in parallel

## Merge adapters after training:
    python scripts/export_merge.py \
    --base_repo google/gemma-3-8b \
    --adapters_dir outputs/gemma3n_qlora_sft \
    --out_dir outputs/gemma3n_qlora_sft_merged

## Notes / Best practices baked in
* QLoRA via Unsloth’s 4-bit load; LoRA attached with PEFT.
* Industry-style configs: one YAML for training; sweep overrides are injected cleanly.
* Packing on SFTTrainer for efficient token use.
* Eval by steps so sweeps get frequent signals.
* Logging to W&B out of the box.
* Export merged weights for fast inference (HF, vLLM, or TGI).
* TF32 enabled in utils for A100 speed.
* FlashAttention2 optional toggle via config.



## UV
* `uv venv`
* `uv pip install XXX`
* `source .venv/bin/activate`
* `uv pip freeze > requirements.txt`
* `uv pip compile requirements.txt -o requirements.lock.txt`
* `uv pip install -r requirements.txt`

## run
* `python3.12 -m  scripts.train --config configs/train.default.yaml`


## Connect
`ssh -i llm-kp.pem ec2-user@[IP-ADDRESS]`


## Setup AWS instance
* `wget -qO- https://astral.sh/uv/install.sh | sh`
* `git clone https://github.com/liamks/fine-tune-llm`
* `cd fine-tune-llm`
* `uv venv`
* `uv pip install torch --index-url https://download.pytorch.org/whl/cu128`
* `uv pip install -r environment/requirements.txt`
* [Attach policy - S3-read-access]
* `aws s3 ls s3://llm-train-data-5ty` # This should work now!
* `aws s3 cp s3://llm-train-data-5ty/ ../data/ --recursive`
* export WANDB_API_KEY=...
* `source .venv/bin/activate`