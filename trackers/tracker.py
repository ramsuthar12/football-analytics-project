from ultralytics import YOLO  # type: ignore
import supervision as sv # type: ignore

class Tracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()

    def detect_frames(self, frames):
        batch_size = 20
        detections = []
        for i in range(0, len(frames), batch_size):
            detections_batch = self.model.predict(frames[i : i+batch_size], conf = 0.1)
            detections += detections_batch 
            break

        return detections

    def get_object_tracker(self, frames):
        detections = self.detect_frames(frames)

        for frame_num, detection in enumerate(detections):
            cls_names = detection.names
            cls_names_inv = {v:k for k,v in cls_names.items()}
            print(cls_names)

            #convert to supervision detection format
            detection_supervision = sv.Detections.from_ultralytics(detection)

            #Convert goalkeeper to player object
            for obj_ind , cls_id in enumerate(detection_supervision.class_id): 
                if cls_names[cls_id] == "goalkeeper":
                    detection_supervision.class_id[obj_ind] = cls_names_inv["player"]

            print(detection_supervision)
            break

