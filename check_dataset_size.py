import os
import json

datasets = ["Ga", "ChapmanShaoxing", "CPSC", "ptbxl"]
splits = ["train", "valid"]

for dataset in datasets:
    for split in splits:
        old_name = f"{dataset}_{split}.jsonl"
        new_name = f"{dataset}_{split}.json"

        # Rename the file if needed
        if os.path.exists(old_name) and not os.path.exists(new_name):
            os.rename(old_name, new_name)

        try:
            with open(new_name, "r") as f:
                data = json.load(f)
                count = len(data)
            print(f"{dataset} {split}: {count} entries")
        except FileNotFoundError:
            print(f"{dataset} {split}: File not found ({new_name})")
        except json.JSONDecodeError:
            print(f"{dataset} {split}: Not a valid JSON file ({new_name})")