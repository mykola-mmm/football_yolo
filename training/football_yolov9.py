

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import utils

import shutil
import wandb
import torch
from wandb.integration.ultralytics import add_wandb_callback
from roboflow import Roboflow
from ultralytics import YOLO


# disable wandb prompt message that requires user input and freezes the kernel
# os.environ['WANDB_MODE'] = 'disabled'

# get api keys
ROBOFLOW_API = utils.get_roboflow_api()
WANDB_API = utils.get_wandb_api()

wandb.login(key=WANDB_API)
wandb.init(project="football_assistant", job_type="training")

rf = Roboflow(api_key=ROBOFLOW_API)
project = rf.workspace("roboflow-jvuqo").project("football-players-detection-3zvbc")
version = project.version(12)
dataset = version.download("yolov9")

# fix data.yaml pathes
utils.fix_dataset_yaml(dataset)

model = YOLO("yolov9m.yaml")
model.info()

device = None
if torch.cuda.is_available():
    print("GPU is available")
    print(f"GPU device name: {torch.cuda.get_device_name(0)}")
    device = 0
else:
    print("GPU is not available")
    device = "cpu"

add_wandb_callback(model, enable_model_checkpointing=True)

results = model.train(data=os.path.join(dataset.location,'data.yaml'),
                      epochs=5,
                      imgsz=640,
                      verbose=True,
                      device=device,
                      project='football_assistant')

wandb.finish()