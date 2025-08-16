import argparse, os
from transformers import TrainingArguments
from trl import SFTTrainer, SFTConfig
from scripts.utils import load_config, deep_update, set_seed, enable_tf32, env_flag
from scripts.dataset import load_jsonl_text_dataset
from scripts.modeling import build_model_and_tokenizer


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--sweep", action="store_true", help="Read overrides from W&B Sweep env")
    return ap.parse_args()


def main():
    args = parse_args()
    base_cfg = load_config(args.config)

    # W&B Sweep: Hydra-like nested overrides come from env vars
    # (W&B sets them in os.environ via the sweep YAML fields)
    # We re-read keys we care about and overlay them.
    sweep_overrides = {}
    for key in [
        "train.learning_rate",
        "train.gradient_accumulation_steps",
        "train.warmup_ration",
        "train.weight_decay",
        "lora.r",
        "lora.alpha",
    ]:
        env_key = key.replace(".", "_").upper()   # e.g. TRAIN_LEARNING_RATE
        if os.getenv(env_key) is not None:
            #place into nested dict
            cur = sweep_overrides
            parts = key.split(".")
            for p in parts[:-1]:
                cur = cur.setdefault(p, {})
            # convert numeric strings to float/int
            val = os.getenv(env_key)
            if val.replace(".","",1).isdigit():
                val = float(val) if "." in val else int(val)
            cur[parts[-1]] = val

    cfg = deep_update(base_cfg, sweep_overrides)

    # Housekeeping
    set_seed(cfg["train"]["seed"])
    enable_tf32()

    # Data
    train_ds = load_jsonl_text_dataset(cfg["data"]["train_path"], cfg["data"]["text_key"])
    eval_ds = load_jsonl_text_dataset(cfg["data"]["eval_path"], cfg["data"]["text_key"])

    # Model
    model, tokenizer = build_model_and_tokenizer(
        base_repo = cfg["model"]["base_repo"],
        max_seq_len = cfg["data"]["max_seq_len"],
        lora_cfg = cfg["lora"],
        attn_impl = cfg["model"]["attn_impl"],
        use_unsloth_compile = cfg["model"]["use_unsloth_compile"],
    )

        # Training args (TRL SFTConfig extends TrainingArguments)
    sft_args = SFTConfig(
        output_dir = cfg["train"]["output_dir"],
        max_steps = cfg["train"]["max_steps"],
        num_train_epochs = cfg["train"]["num_train_epochs"],
        per_device_train_batch_size = cfg["train"]["per_device_train_batch_size"],
        gradient_accumulation_steps = cfg["train"]["gradient_accumulation_steps"],
        eval_strategy = cfg["train"]["eval_strategy"],
        logging_steps = cfg["train"]["logging_steps"],
        eval_steps = cfg["train"]["eval_steps"],
        save_steps = cfg["train"]["save_steps"],
        save_total_limit = cfg["train"]["save_total_limit"],
        warmup_ratio = cfg["train"]["warmup_ratio"],
        learning_rate = cfg["train"]["learning_rate"],
        weight_decay = cfg["train"]["weight_decay"],
        lr_scheduler_type = cfg["train"]["lr_scheduler_type"],
        bf16 = cfg["train"]["bf16"],
        fp16 = cfg["train"]["fp16"],
        gradient_checkpointing = cfg["train"]["gradient_checkpointing"],
        optim = cfg["train"]["optim"],
        report_to = cfg["train"]["report_to"],
        seed = cfg["train"]["seed"],
        run_name = cfg["misc"]["run_name"],
        ddp_find_unused_parameters = False,
    )

    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = train_ds,
        eval_dataset = eval_ds,
        args = sft_args,
        dataset_text_field = cfg["data"]["text_key"],
        packing = True,  # concat samples to fill sequence efficiently
        max_seq_length = cfg["data"]["max_seq_len"],
    )

    trainer.train()
    trainer.save_model(cfg["train"]["output_dir"])  # saves adapters + trait state


if __name__ == "__main__":
    # Optional: avoid any accidental compile at import time in interactive runs
    if env_flag("TORCH_COMPILE_DISABLE_AT_IMPORT", False):
        os.environ["TORCH_COMPILE_DISABLE"] = "1"
    main()
