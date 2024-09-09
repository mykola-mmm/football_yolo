

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import utils

from roboflow import Roboflow
from ultralytics import YOLO
import shutil

# disable wandb prompt message that requires user input and freezes the kernel
os.environ['WANDB_MODE'] = 'disabled'

ROBOFLOW_API = utils.get_roboflow_api()
rf = Roboflow(api_key=ROBOFLOW_API)
project = rf.workspace("roboflow-jvuqo").project("football-players-detection-3zvbc")
version = project.version(12)
dataset = version.download("yolov9")

# fix data.yaml pathes
utils.fix_dataset_yaml(dataset)

# shutil.move('football-players-detection-12/train', 
#             'football-players-detection-12/football-players-detection-12/train')
# shutil.move('football-players-detection-12/test', 
#             'football-players-detection-12/football-players-detection-12/test')
# shutil.move('football-players-detection-12/valid', 
#             'football-players-detection-12/football-players-detection-12/valid')

# model = YOLO("yolov9e.yaml")
model = YOLO("yolov9s.yaml")
model.info()
results = model.train(data=os.path.join(dataset.location,'data.yaml'), epochs=100, imgsz=640, verbose=True, device=0)
