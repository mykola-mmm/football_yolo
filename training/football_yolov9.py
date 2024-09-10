import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import utils

import wandb
import torch
from wandb.integration.ultralytics import add_wandb_callback
from roboflow import Roboflow
from ultralytics import YOLO
from huggingface_hub import HfApi, login


# Get API keys
ROBOFLOW_API = utils.get_roboflow_api()
WANDB_API = utils.get_wandb_api()
HF_TOKEN = utils.get_hugging_face_api()

# Initialize WandB
# os.environ['WANDB_MODE'] = 'disabled'
wandb.require("core")
wandb.login(key=WANDB_API)
wandb.init(project="football_yolov9", job_type="training")
# wandb.init(project="wandb-init", job_type="training",resume="allow", id="seq9v3hx")

# Initialize Hugging Face API

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
add_wandb_callback(model, epoch_logging_interval=3, enable_model_checkpointing=True)

# Train the model
results = model.train(
    data=os.path.join(dataset.location, 'data.yaml'),
    epochs=10,
    batch=-1, # auto mode with 60% GPU utilization
    imgsz=640,
    verbose=True,
    device=devices,
    project='trained_models',
    save_period=4,
    # save=False,
    fraction=0.01,
)

wandb.finish()