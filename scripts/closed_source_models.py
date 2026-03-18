import requests
import json

url = "https://openrouter.ai/api/v1/models"
response = requests.get(url)
data = response.json()

def is_generative_text_model(model):
    """
    Determine if the model is a generative text model (decoder-only or encoder-decoder)
    with text input and text output.
    
    Conditions:
    - Output modalities must include "text".
    - Input modalities must include "text".
    """
    architecture = model.get("architecture", {})
    input_mods = architecture.get("input_modalities", [])
    output_mods = architecture.get("output_modalities", [])
    
    # Must output text
    if "text" not in output_mods:
        return False
    
    # Input must include text
    if "text" not in input_mods:
        return False
    
    return True

models = []
for m in data["data"]:
    model_id = m["id"].lower()

    if not is_generative_text_model(m):
        continue

    # Collect model info
    models.append({
        "model_id": m["id"],
        "provider": "openrouter",
        "architecture": "unknown",
        "context_length": m.get("context_length"),
        "modality": m.get("architecture", {}).get("modality", "unknown")
    })

# Write to JSONL file
with open("closed_models.jsonl", "w") as f:
    for m in models:
        f.write(json.dumps(m) + "\n")

print(f"Number of eligible generative text models: {len(models)}")
print("Sample models:")
for m in models[:5]:
    print(f"  {m['model_id']} (modality: {m['modality']})")
