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
wandb.init(project="wandb-init", job_type="training")

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
if not torch.cuda.is_available():
    devices = "cpu"
    print("Using CPU")
else:
    num_gpus = torch.cuda.device_count()
    if num_gpus == 1:
        devices = 0
        print(f"Using single GPU: {torch.cuda.get_device_name(0)}")
    else:
        devices = 0
        # devices = list(range(num_gpus))
        # print(f"Using {num_gpus} GPUs: {', '.join([torch.cuda.get_device_name(i) for i in range(num_gpus)])}")
print("++++++++++++++++++++++++")
print(devices)
print("++++++++++++++++++++++++")


# # Configure WandB callback
# def wandb_callback(trainer):
#     if trainer.epoch % 10 == 0:
#         # Save model checkpoint
#         checkpoint_path = f"model_checkpoint_epoch_{trainer.epoch}.pt"
#         torch.save(trainer.model.state_dict(), checkpoint_path)
#         wandb.save(checkpoint_path)
        
#         # Log additional visualizations if available
#         if hasattr(trainer, 'validator') and hasattr(trainer.validator, 'metrics'):
#             val_metrics = trainer.validator.metrics
#             if 'confusion_matrix' in val_metrics:
#                 wandb.log({"confusion_matrix": val_metrics['confusion_matrix']})
#             if 'pr_curve' in val_metrics:
#                 wandb.log({"pr_curve": val_metrics['pr_curve']})

# Add custom WandB callback
add_wandb_callback(model, enable_model_checkpointing=True)
# model.add_callback("on_train_epoch_end", wandb_callback)

# Train the model
results = model.train(
    data=os.path.join(dataset.location, 'data.yaml'),
    epochs=100,
    imgsz=640,
    verbose=True,
    device=devices,
    project='wandb-test',
    save_period=5
)

wandb.finish()