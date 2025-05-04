# from models import load_encoder

# encoder, dim = load_encoder("/home/peili/EasyR1/epoch95.pth")

# import torch
# from transformers import AutoConfig
# from verl.models.transformers.time_series_qwen2_5_vl.modeling_time_series_qwen2_5_vl import TimeSeriesQwen2_5_VLForConditionalGeneration
# config = AutoConfig.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
# model = TimeSeriesQwen2_5_VLForConditionalGeneration.from_pretrained(
#     "Qwen/Qwen2.5-VL-7B-Instruct",
#     config=config,
#     torch_dtype=torch.bfloat16,
#     trust_remote_code=True,
# )
# model.time_series_embedding.encoder = encoder

# model.save_pretrained("/home/peili/EasyR1/verl/models/transformers/time_series_qwen2_5_vl", safe_serialization=True)



# import torch
# import os

# local_rank = int(os.environ.get("LOCAL_RANK", 0))  # or use RANK or pass manually
# torch.cuda.set_device(local_rank)

# print("Done!")

# import ray
# ray.init(address="10.1.25.0:6389")

import json
import torch
import os

data_dir = "/scratch/ecg"  # or any of your dataset JSONs
json_path = os.path.join(data_dir, "ChapmanShaoxing_train.json")

with open(json_path, "r") as f:
    data = json.load(f)

missing = 0

for entry in data:
    # print(entry)
    ts_path = entry["time-series"][0]

    full_path = os.path.join(data_dir, ts_path)

    try :
        torch.load(full_path)
    except:
        print(full_path)
        missing += 1
    
    
    
print(f"\n✅ Done checking {len(data)} entries")
print(f"Missing files: {missing}")

# JS01052.pt