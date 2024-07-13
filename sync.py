import classify as classify
from PIL import Image
import cv2

class ObjectIntervalSync:
    def __init__(self, video_path):
        self.classify = classify.Classify()
        self.video_path = video_path
        self.resized_frame = self.classify.process_video(video_path)
        self.frame_values = []
        self.frame_objects = []
        self.frame_scores = []
        self.object_intervals = {obj: [] for obj in classify.ESC_50_classes}
        self.object_status = {obj: {'in_interval': False, 'start_frame': None} for obj in classify.ESC_50_classes}

    def analyze_frames(self, frame_step=2):
        for i, frame in enumerate(self.resized_frame[::frame_step]):
            pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            values, objects = self.classify.recognize_objects(pil_image)
            frame_score = {
                'object': objects,  # top 5
                'score': values
            }
            self.frame_values.append(values)
            self.frame_objects.append(objects)
            self.frame_scores.append(frame_score)
        
    def calculate_intervals(self, frame_rate=30):
        for frame_idx, (objects, scores) in enumerate(zip(self.frame_objects, self.frame_values)):
            for obj, score in zip(objects, scores):
                if score > 0.75:
                    if not self.object_status[obj]['in_interval']:
                        self.object_status[obj]['in_interval'] = True
                        self.object_status[obj]['start_frame'] = frame_idx
                elif score < 0.5:
                    if self.object_status[obj]['in_interval']:
                        self.object_status[obj]['in_interval'] = False
                        start_frame = self.object_status[obj]['start_frame']
                        end_frame = frame_idx
                        duration = (end_frame - start_frame) / frame_rate
                        self.object_intervals[obj].append((start_frame, end_frame, duration))

        # If the video ends and some objects are still in an interval
        for obj, status in self.object_status.items():
            if status['in_interval']:
                start_frame = status['start_frame']
                end_frame = len(self.resized_frame) // 2  # last frame index considered
                duration = (end_frame - start_frame) / frame_rate
                self.object_intervals[obj].append((start_frame, end_frame, duration))

    def get_intervals(self):
        return self.object_intervals
