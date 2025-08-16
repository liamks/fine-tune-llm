import json
from datasets import Dataset


def load_jsonl_text_dataset(path, text_key="text"):
    rows = []
    with open(path, "r") as f:
        for line in f:
            obj = json.loads(line)
            rows.append({"text": obj[text_key]})
    return Dataset.from_list(rows)
