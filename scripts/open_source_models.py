from huggingface_hub import HfApi
import json

TARGET = 12000

api = HfApi()

def is_valid_model(m):
    """Filter unwanted models"""
    # keep only generation models
    if m.pipeline_tag not in ["text-generation", "text2text-generation"]:
        return False

    return True


print("Fetching models from HuggingFace...")

models = api.list_models(
    sort="downloads",
    direction=-1
)

selected = []

for m in models:
    if not is_valid_model(m):
        continue

    entry = {
        "model_id": m.modelId,
        "provider": "huggingface",
        "task": m.pipeline_tag,
        "downloads": m.downloads,
    }

    selected.append(entry)

    if len(selected) >= TARGET:
        break


print(f"Collected {len(selected)} models")

with open("model_registry.jsonl", "w") as f:
    for model in selected:
        f.write(json.dumps(model) + "\n")

print("Saved to model_registry.jsonl")
