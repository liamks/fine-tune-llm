import os, json, hashlib
from pathlib import Path
from typing import Optional
from datasets import Dataset, load_dataset, load_from_disk

def _sanitize_name(name: str) -> str:
    # Keep this short and filesystem-safe
    return "".join(c if c.isalnum() or c in "-._" else "_" for c in name)[:80]

def _default_cache_dir(data_path, tokenizer, max_seq_len, add_eos) -> str:
    # Coerce path-like inputs to strings
    data_path_str = os.fspath(data_path) if isinstance(data_path, (str, os.PathLike)) else str(data_path)

    tok_id = getattr(tokenizer, "name_or_path", "unknown_tokenizer")
    tok_str = os.fspath(tok_id) if isinstance(tok_id, (str, os.PathLike)) else str(tok_id)
    tok_leaf = Path(tok_str).name or "unknown_tokenizer"

    key = f"{os.path.abspath(data_path_str)}|{tok_str}|len={max_seq_len}|eos={int(add_eos)}"
    h = hashlib.sha1(key.encode("utf-8")).hexdigest()[:10]

    base = (
        f".tokcache_{_sanitize_name(Path(data_path_str).stem)}_"
        f"{_sanitize_name(tok_leaf)}_L{max_seq_len}_"
        f"{'eos' if add_eos else 'noeos'}_{h}"
    )
    return str(Path(data_path_str).with_suffix("")) + f"__{base}"


# def _default_cache_dir(data_path: str, tokenizer, max_seq_len: int, add_eos: bool) -> str:
#     # Derive a stable cache dir from file path + tokenizer + settings
#     tok_id = getattr(tokenizer, "name_or_path", "unknown_tokenizer")
#     key = f"{os.path.abspath(data_path)}|{tok_id}|len={max_seq_len}|eos={int(add_eos)}"
#     h = hashlib.sha1(key.encode("utf-8")).hexdigest()[:10]
#     base = f".tokcache_{_sanitize_name(Path(data_path).stem)}_{_sanitize_name(Path(tok_id).split('/')[-1])}_L{max_seq_len}_{'eos' if add_eos else 'noeos'}_{h}"
#     return str(Path(data_path).with_suffix("")) + f"__{base}"

def _write_meta(cache_dir: str, meta: dict):
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(cache_dir) / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

def _read_meta(cache_dir: str) -> Optional[dict]:
    p = Path(cache_dir) / "meta.json"
    if p.exists():
        try:
            return json.load(open(p, "r"))
        except Exception:
            return None
    return None

def load_jsonl_text_dataset(
    path: str,
    text_key: str = "text",
    tokenizer=None,
    max_seq_len: Optional[int] = None,
    cache_dir: Optional[str] = None,
    add_eos: bool = False,
    num_proc: Optional[int] = None,
    map_batch_size: int = 1000,
):
    """
    Behavior:
      - If tokenizer is None or max_seq_len is None -> returns raw text Dataset (backward compatible).
      - Else -> tokenizes once, saves to disk, and reuses cache on subsequent runs.

    Args:
      path: JSONL file path.
      text_key: key in each JSONL line containing the text.
      tokenizer: HF tokenizer instance (e.g., AutoTokenizer.from_pretrained(...)).
      max_seq_len: truncation length for tokenization.
      cache_dir: where to store the tokenized dataset (defaults to a stable path derived from inputs).
      add_eos: append EOS token if not already present (and if tokenizer has eos_token_id).
      num_proc: parallel workers for datasets.map (defaults to datasets’ auto behavior if None).
      map_batch_size: batch size for the batched map (helps throughput).
    """
    # Fallback: just return raw text like your original function
    if tokenizer is None or max_seq_len is None:
        rows = []
        with open(path, "r") as f:
            for line in f:
                obj = json.loads(line)
                rows.append({"text": obj[text_key]})
        return Dataset.from_list(rows)

    # Decide cache location
    cache_dir = cache_dir or _default_cache_dir(path, tokenizer, max_seq_len, add_eos)

    # Check cache validity
    expected_meta = {
        "source_path": os.path.abspath(path),
        "tokenizer_name_or_path": getattr(tokenizer, "name_or_path", "unknown"),
        "max_seq_len": int(max_seq_len),
        "add_eos": bool(add_eos),
        "tokenizer_vocab_size": getattr(tokenizer, "vocab_size", None),
        "eos_token_id": getattr(tokenizer, "eos_token_id", None),
        "format": "hf_datasets_arrow_v1",
        "columns": ["input_ids", "attention_mask"],
    }

    ds_exists = Path(cache_dir).exists()
    meta = _read_meta(cache_dir) if ds_exists else None
    cache_ok = (
        ds_exists
        and meta is not None
        and meta.get("source_path") == expected_meta["source_path"]
        and meta.get("tokenizer_name_or_path") == expected_meta["tokenizer_name_or_path"]
        and meta.get("max_seq_len") == expected_meta["max_seq_len"]
        and meta.get("add_eos") == expected_meta["add_eos"]
    )

    if cache_ok:
        try:
            ds = load_from_disk(cache_dir)
            # quick sanity check for columns
            if all(c in ds.column_names for c in ["input_ids", "attention_mask"]):
                return ds
        except Exception:
            pass  # fall through to re-tokenize

    # No valid cache -> build it
    raw = load_dataset("json", data_files=path, split="train")

    eos_id = getattr(tokenizer, "eos_token_id", None)

    def tokenize_batch(batch):
        texts = batch[text_key]
        out = tokenizer(
            texts,
            truncation=True,
            padding=False,
            max_length=max_seq_len,
        )
        if add_eos and eos_id is not None:
            # Ensure EOS at end if not already present and there's room
            new_input_ids = []
            new_attn = []
            for ids, attn in zip(out["input_ids"], out["attention_mask"]):
                if len(ids) >= max_seq_len:
                    # already truncated; if last isn’t eos, replace last token with eos
                    if ids[-1] != eos_id:
                        ids = ids[:-1] + [eos_id]
                else:
                    # append eos if it's not already present
                    if not ids or ids[-1] != eos_id:
                        ids = ids + [eos_id]
                # rebuild attention mask to match ids length
                attn = [1] * len(ids)
                new_input_ids.append(ids)
                new_attn.append(attn)
            out["input_ids"] = new_input_ids
            out["attention_mask"] = new_attn
        return out

    tokenized = raw.map(
        tokenize_batch,
        batched=True,
        batch_size=map_batch_size,
        num_proc=num_proc,
        remove_columns=[c for c in raw.column_names if c != text_key],
        desc=f"Tokenizing {Path(path).name}",
    )

    # Drop original text column to save space
    if text_key in tokenized.column_names:
        tokenized = tokenized.remove_columns(text_key)

    # Save to disk + meta
    tokenized.save_to_disk(cache_dir)
    _write_meta(cache_dir, expected_meta)

    return tokenized
