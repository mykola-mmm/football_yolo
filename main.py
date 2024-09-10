from utils import *
from trackers import *
import wandb
from ultralytics import YOLO
import supervision as sv
import matplotlib.pyplot as plt
from tqdm import tqdm

SOURCE_VIDEO_PATH = "./input_vids/08fd33_4.mp4"
TARGET_VIDEO_PATH = "./output_vids/08fd33_4_result.mp4"
BALL_ID = 0

def main():
    run = init_wandb(project="wandb-validate", job_type="validation")
    model_path = load_model_and_get_path_wandb(run=run, model_name="mykola-mazniuk-1/wandb-init/run_seq9v3hx_model:latest")
    model = YOLO(model_path)
    print(model.info())



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

    # label_annotator = sv.LabelAnnotator(
    #     color=sv.ColorPalette.DEFAULT,
    #     text_color=sv.Color.from_hex("#000000"),
    # )


    video_info = sv.VideoInfo.from_video_path(SOURCE_VIDEO_PATH)
    video_sink = sv.VideoSink(TARGET_VIDEO_PATH, video_info)
    frame_generator = sv.get_video_frames_generator(SOURCE_VIDEO_PATH)
    frame = next(frame_generator)
    # with video_sink:
    #     for frame in tqdm(frame_generator, total=video_info.total_frames):
    # print(f"model(frame): {model(frame)}")
    result = model(frame)[0]
    # print(f"result: {result}")
    detections = sv.Detections.from_ultralytics(result)
    # print(f"detections: {detections}")
    # print(type(detections))
    ball_detections = detections[detections.class_id == BALL_ID]
    all_detection = detections[detections.class_id != BALL_ID]


    # labels = [
    #     f"{class_name} ({confidence:.2f})"
    #     for class_name, confidence
    #     in zip(detections["class_name"], detections.confidence)
    # ]


    annotated_frame = frame.copy()
    annotated_frame = ellipse_annotator.annotate(annotated_frame, all_detection)
    annotated_frame = triangle_annotator.annotate(annotated_frame, ball_detections)

    # annotated_frame = box_annotator.annotate(annotated_frame, detections)
    # annotated_frame = label_annotator.annotate(annotated_frame, detections, labels)



    # video_sink.write_frame(annotated_frame)

    # sv.plot_image(frame)
    sv.plot_image(annotated_frame)



if __name__ == '__main__':
    main()