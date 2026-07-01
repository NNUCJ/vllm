from modelscope.hub.snapshot_download import snapshot_download

MODEL_NAME = "Qwen/Qwen3.5-35B-A3B"
LOCAL_DIR = "/data/chengjie/models/Qwen/Qwen3.5-35B-A3B"

model_dir = snapshot_download(MODEL_NAME, local_dir=LOCAL_DIR)