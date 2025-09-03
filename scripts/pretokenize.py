# scripts/pretokenize.py
import os, time, json, yaml
from transformers import AutoTokenizer
from pathlib import Path

from scripts.dataset import load_jsonl_text_dataset  # adjust import

def main(cfg_path: str):
    t0 = time.time()
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    base_repo   = cfg["model"]["base_repo"]
    text_key    = cfg["data"]["text_key"]
    max_len     = int(cfg["data"]["max_seq_len"])
    train_path  = cfg["data"]["train_path"]
    eval_path   = cfg["data"]["eval_path"]
    add_eos     = bool(cfg.get("data", {}).get("add_eos", False))

    # Tokenizer only (no model)
    tokenizer = AutoTokenizer.from_pretrained(base_repo, use_fast=True)

    # Speed up map()
    num_proc = os.cpu_count()
    print(f"[pretokenize] Using num_proc={num_proc}, max_len={max_len}")

    # Train
    t1 = time.time()
    train_ds = load_jsonl_text_dataset(
        path=train_path,
        text_key=text_key,
        tokenizer=tokenizer,
        max_seq_len=max_len,
        add_eos=add_eos,
        num_proc=num_proc,
        map_batch_size=1000,
    )
    print(f"[pretokenize] Train cached with columns: {train_ds.column_names}")

    # Eval (optional)
    if Path(eval_path).exists():
        eval_ds = load_jsonl_text_dataset(
            path=eval_path,
            text_key=text_key,
            tokenizer=tokenizer,
            max_seq_len=max_len,
            add_eos=add_eos,
            num_proc=num_proc,
            map_batch_size=1000,
        )
        print(f"[pretokenize] Eval cached with columns: {eval_ds.column_names}")

    print(f"[pretokenize] Done in {time.time()-t0:.1f}s")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    main(args.config)
