from utils import *
from trackers import *
import wandb
from ultralytics import YOLO
import supervision

def main():
    run = init_wandb(project="wandb-validate", job_type="validation")
    model_path = load_model_and_get_path_wandb(run=run, model_name="mykola-mazniuk-1/wandb-init/run_seq9v3hx_model:latest")
    model = YOLO(model_path)
    print(model.info())

if __name__ == '__main__':
    main()