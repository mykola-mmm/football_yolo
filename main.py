import logging
import numpy as np
from utils import *
from trackers import *
import wandb
from ultralytics import YOLO
import supervision as sv
import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2
import datetime
import torch
from transformers import AutoProcessor, SiglipVisionModel
from ids import *
from classifiers import TeamClassifier, goalkeeper_classifier



SIGLIP_MODEL_PATH = "google/siglip-base-patch16-224"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SOURCE_VIDEO_PATH = "./input_vids/08fd33_4.mp4"
TARGET_VIDEO_PATH = "./output_vids/08fd33_4_result.mp4"
TARGET_IMAGE_PATH = f"./output_images/{datetime.datetime.now().strftime('%H%M%S_%Y%m%d')}.jpg"



STRIDE = 30
BATCH_SIZE = 32

PROCESS_FULL_VIDEO = True

def setup_logger():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s\n')
    return logging.getLogger(__name__)

def prepare_siglip_model():
    global SIGLIP_MODEL_PATH, DEVICE, EMBEDDING_MODEL, EMBEDDING_PROCESSOR
    SIGLIP_MODEL_PATH = "google/siglip-base-patch16-224"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    EMBEDDING_MODEL = SiglipVisionModel.from_pretrained(SIGLIP_MODEL_PATH).to(DEVICE)
    EMBEDDING_PROCESSOR = AutoProcessor.from_pretrained(SIGLIP_MODEL_PATH)

def extract_crops(model: YOLO, source_video_path, stride=STRIDE):
    frame_generator = sv.get_video_frames_generator(source_video_path, stride=stride)
    crops = []

    video_info = sv.VideoInfo.from_video_path(source_video_path)

    
    for frame in tqdm(frame_generator, total=video_info.total_frames//stride, desc="Collecting crops"):
        result = model(frame, conf=0.3)[0]
        detections = sv.Detections.from_ultralytics(result)
        detections = detections.with_nms(threshold=0.5, class_agnostic=False)
        detections = detections[detections.class_id == PLAYER_ID]
        crops += [
            sv.crop_image(frame, xyxy) for xyxy in detections.xyxy
        ]
    return crops

def main():
    os.makedirs('output_vids', exist_ok=True)
    logger =setup_logger()
    model_path = './models/yolov9m_epoch284.pt'
    model = YOLO(model_path)
    pitch_detection_model = "./models/pitch_keypoints_detection_yolov8x.pt"
    pitch_model = YOLO(pitch_detection_model)
    print(model.info())
    print(pitch_model.info())

    # initialize team classifier
    team_classifier = TeamClassifier(model_path, SIGLIP_MODEL_PATH)
    team_classifier.fit(source_video_path=SOURCE_VIDEO_PATH,
                        stride=STRIDE,
                        confidence_threshold=0.3,
                        umap_n_components=3,
                        kmeans_n_clusters=2,
                        batch_size=32,
                        debug=False)

    ellipse_annotator = sv.EllipseAnnotator(
        color=sv.ColorPalette.DEFAULT,
        thickness=2,
    )

    triangle_annotator = sv.TriangleAnnotator(
        color=sv.Color.from_hex("#FFD700"),
        base=20, height=17
    )

    vertex_annotator = sv.VertexAnnotator(
        color=sv.Color.from_hex("#FFD700"),
        radius=10,
    )

    label_annotator = sv.LabelAnnotator(
        color=sv.ColorPalette.DEFAULT,
        text_color=sv.Color.from_hex("#000000"),
        text_position=sv.Position.BOTTOM_CENTER,
    )

    tracker = sv.ByteTrack()
    tracker.reset()


    video_info = sv.VideoInfo.from_video_path(SOURCE_VIDEO_PATH)
    video_sink = sv.VideoSink(TARGET_VIDEO_PATH, video_info)
    frame_generator = sv.get_video_frames_generator(SOURCE_VIDEO_PATH)
    frame = next(frame_generator)
    i = 1
    with video_sink:
        for frame in tqdm(frame_generator, total=video_info.total_frames):

            result = model(frame)[0]
            detections = sv.Detections.from_ultralytics(result)
            ball_detections = detections[detections.class_id == BALL_ID]
            ball_detections.xyxy = sv.pad_boxes(ball_detections.xyxy, px=10)
            all_detections = detections[detections.class_id != BALL_ID]

            all_detections = all_detections.with_nms(threshold=0.5, class_agnostic=False)
            all_detections = tracker.update_with_detections(detections=all_detections)

            player_detections = all_detections[all_detections.class_id == PLAYER_ID]
            player_crops = [sv.crop_image(frame, xyxy) for xyxy in player_detections.xyxy]
            # logger.info(f"player_detections.class_id: {player_detections.class_id}")
            player_detections.class_id = team_classifier.predict(player_crops)
            # logger.info(f"player_detections.class_id: {player_detections.class_id}")

            goalkeeper_detections = all_detections[all_detections.class_id == GOALKEEPER_ID]
            goalkeeper_detections.class_id = goalkeeper_classifier(player_detections, goalkeeper_detections)

            referee_detections = all_detections[all_detections.class_id == REFEREE_ID]

            # There is an issue with empty goalkeeper_detections
            # all_detections = sv.Detections.merge([player_detections, goalkeeper_detections, referee_detections])

            all_detections = sv.Detections(
                xyxy=np.empty((0, 4), dtype=np.float32),
                confidence=np.array([], dtype=np.float32),
                class_id=np.array([], dtype=int)
            )

            if player_detections.class_id.size != 0:
                all_detections = sv.Detections.merge([all_detections, player_detections])
            if goalkeeper_detections.class_id.size != 0:
                all_detections = sv.Detections.merge([all_detections, goalkeeper_detections])
            if referee_detections.class_id.size != 0:
                all_detections = sv.Detections.merge([all_detections, referee_detections])

            labels = [
                f"#{tracker_id}"
                for tracker_id in all_detections.tracker_id
            ]

            result = pitch_model(frame, conf=0.3)[0]
            key_points = sv.KeyPoints.from_ultralytics(result)

            annotated_frame = frame.copy()
            annotated_frame = ellipse_annotator.annotate(annotated_frame, all_detections)
            annotated_frame = triangle_annotator.annotate(annotated_frame, ball_detections)

            annotated_frame = label_annotator.annotate(annotated_frame, all_detections, labels)
            annotated_frame = vertex_annotator.annotate(annotated_frame, key_points)

            if PROCESS_FULL_VIDEO:
                video_sink.write_frame(annotated_frame)
            else:
                sv.plot_image(annotated_frame)
                cv2.imwrite(TARGET_IMAGE_PATH, annotated_frame)
                break



if __name__ == '__main__':
    main()
