import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import utils

import wandb
import torch
from wandb.integration.ultralytics import add_wandb_callback
from roboflow import Roboflow
from ultralytics import YOLO

# Get API keys
ROBOFLOW_API = utils.get_roboflow_api()
WANDB_API = utils.get_wandb_api()

# Initialize WandB
wandb.require("core")
wandb.login(key=WANDB_API)
# wandb.init(project="wandb-init", job_type="training")
wandb.init(project="wandb-init", job_type="training",resume="allow", id="run_seq9v3hx_model")


# Set up Roboflow dataset
rf = Roboflow(api_key=ROBOFLOW_API)
project = rf.workspace("roboflow-jvuqo").project("football-players-detection-3zvbc")
version = project.version(12)
dataset = version.download("yolov9")

# Fix data.yaml paths
utils.fix_dataset_yaml(dataset)

# Initialize YOLO model
model = YOLO("yolov9m.pt")
model.info()

# Set device based on GPU availability
devices = None
try:
    if not torch.cuda.is_available():
        devices = "cpu"
        print("Using CPU")
    else:
        num_gpus = torch.cuda.device_count()
        if num_gpus == 1:
            devices = 0
            print(f"Using single GPU: {torch.cuda.get_device_name(0)}")
        else:
            devices = list(range(num_gpus))
            print(f"Using {num_gpus} GPUs: {', '.join([torch.cuda.get_device_name(i) for i in range(num_gpus)])}")
finally:
    print("\n++++++++++++++++++++++++")
    print(f"Devices: {devices}")
    print("++++++++++++++++++++++++\n")


# Add custom WandB callback
add_wandb_callback(model, enable_model_checkpointing=True)

# Train the model
results = model.train(
    data=os.path.join(dataset.location, 'data.yaml'),
    epochs=100,
    batch=-1, # auto mode with 60% GPU utilization
    imgsz=1280,
    verbose=True,
    device=devices,
    project='wandb-test',
    save_period=5,
)

wandb.finish()