from utils import *
from trackers import *
import wandb
from ultralytics import YOLO
import supervision as sv

SOURCE_VIDEO_PATH = "./input_vids/08fd33_4.mp4"

def main():
    run = init_wandb(project="wandb-validate", job_type="validation")
    model_path = load_model_and_get_path_wandb(run=run, model_name="mykola-mazniuk-1/wandb-init/run_seq9v3hx_model:latest")
    model = YOLO(model_path)
    print(model.info())
    frame_generator = sv.get_video_frames_generator(SOURCE_VIDEO_PATH)
    frame = next(frame_generator)
    sv.plot_image(frame)

if __name__ == '__main__':
    main()