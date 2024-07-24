import classify as classify
from PIL import Image
import cv2
from collections import Counter

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
        self.weather_counts = Counter()
        self.place_counts = Counter()
        self.scene_counts = Counter()
        self.time_counts = Counter()
        self.ambience = None

    def analyze_frames(self, frame_step=2):
        for i, frame in enumerate(self.resized_frame[::frame_step]):
            pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            image_tensor = self.classify.preprocess_image(pil_image)

            values, objects = self.classify.recognize_objects(pil_image)
            frame_score = {
                'object': objects,  # top 5
                'score': values
            }
            self.frame_values.append(values)
            self.frame_objects.append(objects)
            self.frame_scores.append(frame_score)

            # Classify weather, place, and scene
            weather = self.classify.classify_weather(image_tensor)
            place = self.classify.classify_place(image_tensor)
            scene = self.classify.classify_scene(image_tensor)
            scene = self.classify.classify_time(image_tensor)

            self.weather_counts[weather] += 1
            self.place_counts[place] += 1
            self.scene_counts[scene] += 1
            self.time_counts[scene] += 1

        self.determine_ambience()

    def determine_ambience(self):
        weather = self.weather_counts.most_common(1)[0][0]
        place = self.place_counts.most_common(1)[0][0]
        scene = self.scene_counts.most_common(1)[0][0]
        time = self.time_counts.most_common(1)[0][0]

        if weather == "windy":
            self.ambience = "windy"
        elif weather == "thunderstorm":
            self.ambience = "thunderstorm"
        elif weather == "rainy":
            self.ambience = "rainy"
        elif weather == "drizzle":
            self.ambience = "drizzle"
        elif weather == "sunny":
            if place == "nature":
                if time == "day":
                    self.ambience = "nature day"
                else:
                    self.ambience = "nature night"
            elif place == "urban":
                if scene == "indoors":
                    self.ambience = "urban indoor"
                else:
                    self.ambience = "urban outdoor"

        
    def calculate_intervals(self, frame_rate=30):
        for frame_idx, (objects, scores) in enumerate(zip(self.frame_objects, self.frame_values)):
            for obj, score in zip(objects, scores):
                if score > 0.75:
                    if not self.object_status[obj]['in_interval']:
                        self.object_status[obj]['in_interval'] = True
                        self.object_status[obj]['start_frame'] = frame_idx
                elif score < 0.3:
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
        self.object_intervals = {k: v for k, v in self.object_intervals.items() if v}
        return self.object_intervals
    
    def get_ambience(self):
        return self.ambience
