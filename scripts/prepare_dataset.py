# scripts/prepare_dataset.py
import json, argparse, random
from transformers import AutoTokenizer

def to_text(user_prompt, assistant_response, eos=""):
    """
    Format into Gemma's expected 2-role style:
    <start_of_turn>user
    [full user content]<end_of_turn>
    <start_of_turn>model
    [assistant content]<end_of_turn>
    """
    return (
        f"<start_of_turn>user\n"
        f"{user_prompt.strip()}\n<end_of_turn>\n"
        f"<start_of_turn>model\n"
        f"{assistant_response.strip()}{eos}<end_of_turn>"
    )

def convert(in_path, out_path, base_repo=None, sample_frac=1.0, seed=3407):
    """
    Reads a JSONL with keys: system, user, assistant
    Outputs JSONL with a single 'text' field in Gemma's format
    """
    rng = random.Random(seed)

    # EOS token if you want to include it after assistant output
    eos_token = ""
    if base_repo:
        try:
            tok = AutoTokenizer.from_pretrained(base_repo, use_fast=True)
            eos_token = tok.eos_token or ""
        except Exception:
            pass

    with open(out_path, "w") as out_f, open(in_path, "r") as in_f:
        for line in in_f:
            if not line.strip():
                continue
            ex = json.loads(line)

            # Merge system + user into one "user" turn
            if ex.get("system"):
                merged_user = f"{ex['system'].strip()}\n\n{ex['user'].strip()}"
            else:
                merged_user = ex["user"].strip()

            text = to_text(merged_user, ex["assistant"], eos=eos_token)

            if sample_frac < 1.0 and rng.random() > sample_frac:
                continue

            out_f.write(json.dumps({"text": text}) + "\n")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="input jsonl with system,user,assistant")
    ap.add_argument("--out", dest="out", required=True, help="output jsonl with text")
    ap.add_argument("--base_repo", help="base model repo (to get EOS token)")
    ap.add_argument("--sample_frac", type=float, default=1.0)
    args = ap.parse_args()

    convert(args.inp, args.out, base_repo=args.base_repo, sample_frac=args.sample_frac)
