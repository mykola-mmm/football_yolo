from ultralytics import YOLO
import supervision as sv

class Tracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTracker()

    def get_object_tracks(self, frames):
        detections = self.detect_frames(frames)
        for player_num, detections in enumerate(detections):
            for detection in detections:
                cls_name = detections.names
                cls_names_inversed = {v:k for k, v in detections.names.items()}

                detection_supervision = sv.Detections.from_ultralytics(detection)

    def detect_frames(self, frames, batch_size=10, conf=0.1):
        detections = []
        for i in range(0, len(frames), batch_size):
            detections_bacth = self.model.predict(frames[i:i+batch_size],conf=conf)
            detections += detections_bacth
        return detections