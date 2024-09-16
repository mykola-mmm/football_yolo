# football_yolo

This project focuses on detecting and classifying football players, goalkeepers, and referees in video footage using YOLO (You Only Look Once) models and various machine learning techniques.

## What Was Done
- **Player Detection**: Utilized YOLO models to detect players, goalkeepers, ball and referees in video frames.
- **Team Classification**: Implemented a team classifier using SiglipVisionModel and UMAP for dimensionality reduction and KMeans for clustering.
- **Training**: Trained custom YOLO models for pitch and player detection using datasets from Roboflow.

## Technological Stack
- **YOLO**: Used for object detection.
- **Transformers**: SiglipVisionModel for feature extraction.
- **UMAP**: For dimensionality reduction.
- **KMeans**: For clustering player embeddings.
- **OpenCV**: For video processing.
- **WandB**: For experiment tracking and model checkpointing.
- **Roboflow**: For dataset management.
- **Supervision**: For handling detections and annotations.

## Demo Video

![Football GIF](./input_vids/08fd33_4-ezgif.com-video-to-gif-converter.gif)
![Football GIF](./output_vids/08fd33_4_result-ezgif.com-video-to-gif-converter.gif)

