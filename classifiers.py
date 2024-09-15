from transformers import AutoProcessor, SiglipVisionModel
import torch
from ultralytics import YOLO
from tqdm import tqdm
import supervision as sv
from ids import *
from more_itertools import chunked
import numpy as np
import umap
from sklearn.cluster import KMeans

class TeamClassifier:
    def __init__(self, yolo_model_path, siglip_model_path):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.siglip_processor = AutoProcessor.from_pretrained(siglip_model_path)
        self.siglip_model = SiglipVisionModel.from_pretrained(siglip_model_path).to(self.device)
        self.yolo_model = YOLO(yolo_model_path)

    def fit(self, source_video_path, stride, confidence_threshold, umap_n_components=3, kmeans_n_clusters=2, batch_size=32, debug=False):
        self.source_video_path = source_video_path
        self.stride = stride
        self.confidence_threshold = confidence_threshold
        self.batch_size = batch_size
        self.umap_n_components = umap_n_components
        self.kmeans_n_clusters = kmeans_n_clusters
        crops = self.extract_crops(self.source_video_path, self.stride, self.confidence_threshold)
        if debug:
            sv.plot_images_grid(crops[:100], grid_size=(10, 10))

        crops = [sv.cv2_to_pillow(crop) for crop in crops]
        batches = chunked(crops, self.batch_size)
        data = []

        with torch.no_grad():
            for batch in tqdm(batches, total=len(crops)//self.batch_size, desc="Embeddings extraction"):
                inputs = self.siglip_processor(images=batch, return_tensors="pt").to(self.device)
                outputs = self.siglip_model(**inputs)
                embeddings = torch.mean(outputs.last_hidden_state, dim=1).cpu().numpy()
                data.append(embeddings)

        data = np.concatenate(data)

        self.umap_reducer = umap.UMAP(n_components=self.umap_n_components)
        self.clustering_model = KMeans(n_clusters=self.kmeans_n_clusters)

        projections = self.umap_reducer.fit_transform(data)
        self.clustering_model.fit(projections)


    def predict(self, crops):
        crops = [sv.cv2_to_pillow(crop) for crop in crops]
        batches = chunked(crops, self.batch_size)
        data = []

        with torch.no_grad():
            for batch in tqdm(batches, total=len(crops)//self.batch_size, desc="Embeddings extraction"):
                inputs = self.siglip_processor(images=batch, return_tensors="pt").to(self.device)
                outputs = self.siglip_model(**inputs)
                embeddings = torch.mean(outputs.last_hidden_state, dim=1).cpu().numpy()
                data.append(embeddings)

        data = np.concatenate(data)
        projections = self.umap_reducer.transform(data)
        return self.clustering_model.predict(projections)

    def extract_crops(self, source_video_path, stride, confidence_threshold):
        frame_generator = sv.get_video_frames_generator(source_video_path, stride=stride)
        crops = []
        video_info = sv.VideoInfo.from_video_path(source_video_path)
        for frame in tqdm(frame_generator, total=video_info.total_frames//stride, desc="Collecting crops"):
            result = self.yolo_model(frame, conf=confidence_threshold)[0]
            detections = sv.Detections.from_ultralytics(result)
            detections = detections.with_nms(threshold=0.5, class_agnostic=False)
            detections = detections[detections.class_id == PLAYER_ID]
            crops += [
                sv.crop_image(frame, xyxy) for xyxy in detections.xyxy
            ]
        return crops


def goalkeeper_classifier(player_detections: sv.Detections,
                          goalkeeper_detections: sv.Detections):
    goalkeepers_xy = goalkeeper_detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
    players_xy = player_detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)

    team_0_centroid = players_xy[player_detections.class_id == 0].mean(axis=0)
    team_1_centroid = players_xy[player_detections.class_id == 1].mean(axis=0)

    goalkeepers_team_ids = []
    for goalkeeper_xy in goalkeepers_xy:
        distances_to_team_0 = np.linalg.norm(goalkeeper_xy - team_0_centroid)
        distances_to_team_1 = np.linalg.norm(goalkeeper_xy - team_1_centroid)
        goalkeepers_team_ids.append(0 if distances_to_team_0 < distances_to_team_1 else 1)

    return np.array(goalkeepers_team_ids)
    
























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












        

# def extract_crops(model: YOLO, source_video_path, stride=STRIDE):
#     frame_generator = sv.get_video_frames_generator(source_video_path, stride=stride)
#     crops = []

#     video_info = sv.VideoInfo.from_video_path(source_video_path)

    
#     for frame in tqdm(frame_generator, total=video_info.total_frames//stride, desc="Collecting crops"):
#         result = model(frame, conf=0.3)[0]
#         detections = sv.Detections.from_ultralytics(result)
#         detections = detections.with_nms(threshold=0.5, class_agnostic=False)
#         detections = detections[detections.class_id == PLAYER_ID]
#         crops += [
#             sv.crop_image(frame, xyxy) for xyxy in detections.xyxy
#         ]
#     return crops