import json
import os
from insightface.app import FaceAnalysis

MODEL_NAME = "buffalo_l"
THRESHOLD = 0.55

model = FaceAnalysis(name=MODEL_NAME, providers=["CPUExecutionProvider"])
model.prepare(ctx_id=-1, det_size=(640, 640))


def save_config(output_dir="model"):
    os.makedirs(output_dir, exist_ok=True)
    config = {
        "model_name": MODEL_NAME,
        "threshold": THRESHOLD,
        "det_size": [640, 640],
    }
    path = os.path.join(output_dir, "config.json")
    with open(path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Config saved → {path}")


if __name__ == "__main__":
    print(f"Model ready: {MODEL_NAME}")
    save_config()
    print("Done.")