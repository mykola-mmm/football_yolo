from utils import *
from trackers import *
import wandb
from ultralytics import YOLO

def main():
    ROBOFLOW_API = utils.get_roboflow_api()
    WANDB_API = utils.get_wandb_api()
    wandb.require("core")
    wandb.login(key=WANDB_API)
    run = wandb.init(project="wandb-validate", job_type="validation")
    model_path = run.use_model(name="mykola-mazniuk-1/wandb-init/run_seq9v3hx_model:latest")
    print(model_path)
    model = YOLO(model_path)
    pass 

if __name__ == '__main__':
    main()