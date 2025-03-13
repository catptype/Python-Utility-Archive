import numpy as np

from typing import Any, Tuple
from ultralytics import YOLO
from ultralytics.engine.results import Results

class YoloObjectDetector:
    def __init__(self, model):
        self.model = YOLO(model, task='detect')
        self.predict(np.zeros((1, 1, 3), dtype=np.uint8))
    
    # --- Methods ---
    def get_speed(self, results:Results) -> dict:
        return results.speed

    def get_image_size(self, results:Results) -> Tuple[int, int]:
        return results.orig_shape
    
    def get_info(self, results:Results, mode='xyxy') -> list:
        if not mode in ['xyxy', 'xyxyn', 'xywh', 'xywhn']:
            raise ValueError(f"mode must be ['xyxy', 'xyxyn', 'xywh', 'xywhn']")
        
        boxes = results.boxes

        box = [[int(val) for val in box] for box in getattr(boxes, mode).tolist()] if mode in ['xyxy', 'xywh'] else getattr(boxes, mode).tolist()
        score = boxes.conf.tolist()
        cls_name = [self.cls_dict[id] for id in boxes.cls.tolist()]

        if boxes.is_track:
            track_id = map(int, boxes.id.tolist())
            return zip(track_id, box, score, cls_name)

        else:
            return zip(box, score, cls_name)

    def tracking(self, source, conf:float=0.5, iou:float=0.5, max_det=100, agnostic_nms=True, **kwargs:Any) -> Results:
        """
        Ultralytic defaults
        max_det=300
        agnostic_nms=False
        """
        return self.model.track(source, conf=conf, iou=iou, max_det=max_det, agnostic_nms=agnostic_nms, verbose=False, **kwargs)[0]
    
    def predict(self, source, conf:float=0.5, iou:float=0.5, max_det=100, agnostic_nms=True, **kwargs:Any) -> Results:
        """
        Ultralytic defaults
        max_det=300
        agnostic_nms=False
        """
        return self.model(source, conf=conf, iou=iou, max_det=max_det, agnostic_nms=agnostic_nms, verbose=False, **kwargs)[0]
    
    # Extract image
    def extract_detect(self, image:np.ndarray, box_xyxy:tuple, offset:int = 0) -> np.ndarray:

        height, width = image.shape[:2]

        # Decode box coordinate tuple
        x1, y1, x2, y2 = box_xyxy

        # Processing coordinate with offset
        x1 = max(x1 - offset, 0) # prevent x1 become negative
        y1 = max(y1 - offset, 0) # prevent y1 become negative
        x2 = min(width, x2 + offset) # prevent x2 overflow from vdeo frame
        y2 = min(height, y2 + offset) # prevent y2 overflow from vdeo frame

        # Extract image
        extract_image = image[y1:y2, x1:x2]
        return extract_image
    
    # --- Properties ---
    @property
    def cls_names(self) -> list:
        cls_dict:dict = self.model.names
        return list(cls_dict.values())
    
    @property
    def cls_dict(self) -> dict:
        return self.model.names