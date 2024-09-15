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
from more_itertools import chunked
import umap
from sklearn.cluster import KMeans
from ids import *
from team_classifier import TeamClassifier



SIGLIP_MODEL_PATH = "google/siglip-base-patch16-224"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SOURCE_VIDEO_PATH = "./input_vids/08fd33_4.mp4"
TARGET_VIDEO_PATH = "./output_vids/08fd33_4_result.mp4"
TARGET_IMAGE_PATH = f"./output_images/{datetime.datetime.now().strftime('%H%M%S_%Y%m%d')}.jpg"



STRIDE = 30
BATCH_SIZE = 32

TEST = True

def setup_logger():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
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
    logger =setup_logger()
    # run = init_wandb(project="wandb-validate", job_type="validation")
    # model_path = load_model_and_get_path_wandb(run=run, model_name="mykola-mazniuk-1/wandb-init/run_seq9v3hx_model:latest")
    model_path = './models/yolov9m_epoch284.pt'
    model = YOLO(model_path)
    print(model.info())

    # initialize team classifier
    team_classifier = TeamClassifier(model_path, SIGLIP_MODEL_PATH)
    team_classifier.fit(source_video_path=SOURCE_VIDEO_PATH,
                        stride=STRIDE,
                        confidence_threshold=0.3,
                        umap_n_components=3,
                        kmeans_n_clusters=2,
                        batch_size=32,
                        debug=False)

    if TEST:
        ellipse_annotator = sv.EllipseAnnotator(
            color=sv.ColorPalette.DEFAULT,
            thickness=2,
        )

        triangle_annotator = sv.TriangleAnnotator(
            color=sv.Color.from_hex("#FFD700"),
            base=20, height=17
        )

        # box_annotator = sv.BoxAnnotator(
        #     color=sv.ColorPalette.DEFAULT,
        #     thickness=2,
        # )

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
        # with video_sink:
        #     for frame in tqdm(frame_generator, total=video_info.total_frames):
        # print(f"model(frame): {model(frame)}")
        raw_result = model(frame)
        # logger.info(f"type(raw_result): {type(raw_result)}")

        result = model(frame)[0]
        # print(f"result: {result}")
        detections = sv.Detections.from_ultralytics(result)
        # print(f"detections: {detections}")
        # print(type(detections))
        ball_detections = detections[detections.class_id == BALL_ID]
        ball_detections.xyxy = sv.pad_boxes(ball_detections.xyxy, px=10)
        all_detections = detections[detections.class_id != BALL_ID]

        all_detections = all_detections.with_nms(threshold=0.5, class_agnostic=False)
        # all_detections.class_id = all_detections.class_id - 1
        logger.info(f"all_detections.tracker_id: {all_detections.tracker_id}")
        all_detections = tracker.update_with_detections(detections=all_detections)

        player_detections = all_detections[all_detections.class_id == PLAYER_ID]
        player_crops = [sv.crop_image(frame, xyxy) for xyxy in player_detections.xyxy]
        logger.info(f"player_detections.class_id: {player_detections.class_id}")
        player_detections.class_id = team_classifier.predict(player_crops)
        logger.info(f"player_detections.class_id: {player_detections.class_id}")

        labels = [
            f"#{tracker_id}"
            for tracker_id in player_detections.tracker_id
        ]
        logger.info(f"labels: {labels}")


        annotated_frame = frame.copy()
        annotated_frame = ellipse_annotator.annotate(annotated_frame, player_detections)
        annotated_frame = triangle_annotator.annotate(annotated_frame, ball_detections)

        # annotated_frame = box_annotator.annotate(annotated_frame, detections)
        annotated_frame = label_annotator.annotate(annotated_frame, player_detections, labels)



        # video_sink.write_frame(annotated_frame)

        # sv.plot_image(frame)
        sv.plot_image(annotated_frame)
        cv2.imwrite(TARGET_IMAGE_PATH, annotated_frame)
        # cv2.imwrite(TARGET_IMAGE_PATH, cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR))

    # if not TEST:
    #     crops = extract_crops(model, SOURCE_VIDEO_PATH, stride=STRIDE)
    #     sv.plot_images_grid(crops[:100], grid_size=(10, 10))

    #     prepare_siglip_model()

    #     crops = [sv.cv2_to_pillow(crop) for crop in crops]
    #     logger.info(f"len(crops): {len(crops)}")

    #     batches = chunked(crops, BATCH_SIZE)

    #     data = []

    #     with torch.no_grad():
    #         for batch in tqdm(batches, desc="Embeddings extraction"):
    #             inputs = EMBEDDING_PROCESSOR(images=batch, return_tensors="pt").to(DEVICE)
    #             outputs = EMBEDDING_MODEL(**inputs)
    #             logger.info(f"outputs.last_hidden_state.shape: {outputs.last_hidden_state.shape}")
    #             embeddings = torch.mean(outputs.last_hidden_state, dim=1).cpu().numpy()
    #             logger.info(f"embeddings.shape: {embeddings.shape}")
    #             data.append(embeddings)

    #     data = np.concatenate(data)
    #     logger.info(f"data.shape: {data.shape}")

    #     REDUCER = umap.UMAP(n_components=3)
    #     CLUSTERING_NODEL = KMeans(n_clusters=2)

    #     projections = REDUCER.fit_transform(data)
    #     logger.info(f"projections.shape: {projections.shape}")

    #     clusters = CLUSTERING_NODEL.fit_predict(projections)
    #     logger.info(f"clusters: {clusters}")

    #     team_0 = [
    #         crop
    #         for crop, cluster
    #         in zip(crops, clusters)
    #         if cluster == 0
    #     ]

    #     team_1 = [
    #         crop
    #         for crop, cluster
    #         in zip(crops, clusters)
    #         if cluster == 1
    #     ]

    #     sv.plot_images_grid(team_0[:100], grid_size=(10, 10))
    #     sv.plot_images_grid(team_1[:100], grid_size=(10, 10))


if __name__ == '__main__':
    main()
