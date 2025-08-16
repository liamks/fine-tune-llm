import os, argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_repo", required=True)
    ap.add_argument("--adapters_dir", required=True)  # outputs/... directory
    ap.add_argument("--out_dir", required=True)
    return ap.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    base = AutoModelForCausalLM.from_pretrained(
        args.base_repo, device_map="cpu", torch_dtype=torch.bfloat16
    )

    model = PeftModel.from_pretrained(base, args.adapters_dir)
    model = model.merge_and_unload()
    model.save_pretrained(args.out_dir, safe_serialization=True)
    tokenizer = AutoTokenizer.from_pretrained(args.base_repo)
    tokenizer.save_pretrained(args.out_dir)
    print(f"Merged model saved to {args.out_dir}")


if __name__ == "__main__":
    main()