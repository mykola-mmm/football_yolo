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
wandb.init(project="football_assistant", job_type="training")

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

# Set device
device = None
if torch.cuda.is_available():
    print("GPU is available")
    print(f"GPU device name: {torch.cuda.get_device_name(0)}")
    device = 0
else:
    print("GPU is not available")
    device = "cpu"

# Configure WandB callback
def wandb_callback(trainer):
    if trainer.epoch % 5 == 0:
        # Save model checkpoint
        checkpoint_path = f"model_checkpoint_epoch_{trainer.epoch}.pt"
        trainer.model.save(checkpoint_path)
        wandb.save(checkpoint_path)
        
        # Log additional visualizations
        wandb.log({
            "pr_curve": wandb.plot.pr_curve(trainer.validation.probs, trainer.validation.targets, labels=trainer.model.names),
            "confusion_matrix": wandb.plot.confusion_matrix(probs=trainer.validation.probs, y_true=trainer.validation.targets, class_names=trainer.model.names),
        })

# Add custom WandB callback
add_wandb_callback(model, enable_model_checkpointing=True)
model.add_callback("on_train_epoch_end", wandb_callback)

# Train the model
results = model.train(
    data=os.path.join(dataset.location, 'data.yaml'),
    epochs=100,
    imgsz=1280,
    verbose=True,
    device=device,
    project='football_assistant'
)

wandb.finish()