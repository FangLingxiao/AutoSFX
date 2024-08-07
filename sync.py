import torch
import classify as classify
from PIL import Image
import cv2
from collections import Counter
import numpy as np

class ObjectIntervalSync:
    def __init__(self, video_path, output_dir='grad_cam_heatmaps', save_interval=15):
        self.video_path = video_path
        self.fps = self.get_video_fps()
        self.classify = classify.Classify()
        self.resized_frame = self.classify.process_video(video_path, output_dir=output_dir)
        self.frame_values = []
        self.frame_objects = []
        self.frame_scores = []
        self.object_intervals = {obj: [] for obj in classify.ESC_50_classes}
        self.object_status = {obj: {'in_interval': False, 'start_frame': None, 'needs_fine_sync': False} for obj in classify.ESC_50_classes}
        self.weather_counts = Counter()
        self.place_counts = Counter()
        self.scene_counts = Counter()
        self.time_counts = Counter()
        self.ambience = None
        self.object_infos = []
        self.fine_sync_objects = [
            "thunderstorm", "door_wood_knock", "can_opening", "clapping",
            "mouse_click", "water_drops", "clock_alarm", 
            "footsteps walking running", "glass_breaking"
        ]
        self.start_threshold = 0.65
        self.end_threshold = 0.35
        self.min_interval_duration = 1.0 
        self.window_size = 5 
        self.confidence_history = {obj: [] for obj in self.object_status.keys()}
        self.total_frames = len(self.resized_frame)  # Save the actual total number of frames
        self.end_consecutive_frames = 5  # The number of consecutive frames below the threshold to end the detection

    def get_top_5_objects(self):
        object_counter = Counter()
        for objects in self.frame_objects:
            object_counter.update(objects)
        
        return object_counter.most_common(5)

    def get_video_fps(self):
        video = cv2.VideoCapture(self.video_path)
        if not video.isOpened():
            print(f"Error opening video file: {self.video_path}")
            return 30  # default FPS

        fps = video.get(cv2.CAP_PROP_FPS)
        video.release()

        if fps <= 0:
            print(f"Invalid FPS ({fps}) detected, using default 30 FPS")
            return 30
        return fps
    
    def needs_fine_sync(self, obj):
        return obj in self.fine_sync_objects

    def analyze_frames(self, frame_step=2):
        for i, frame in enumerate(self.resized_frame):
        #for i, frame in enumerate(self.resized_frame[::frame_step]):
            pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            image_tensor = self.classify.preprocess_image(pil_image)

            with torch.no_grad():
                values, objects = self.classify.recognize_objects(pil_image)

            # Grad-CAM needs gradience
            heatmap = self.classify.get_grad_cam(pil_image)
            object_info = self.classify.analyze_heatmap(heatmap)

            frame_score = {
                'object': objects,
                'score': values,
                'object_info': object_info
            }
            self.frame_values.append(values)
            self.frame_objects.append(objects)
            self.frame_scores.append(frame_score)
            self.object_infos.append(object_info)

            # Classify weather, place, and scene
            weather = self.classify.classify_weather(image_tensor)
            place = self.classify.classify_place(image_tensor)
            scene = self.classify.classify_scene(image_tensor)
            scene = self.classify.classify_time(image_tensor)

            self.weather_counts[weather] += 1
            self.place_counts[place] += 1
            self.scene_counts[scene] += 1
            self.time_counts[scene] += 1

            #print(f"Frame {i}: Objects: {objects}, Scores: {values}")

        #for obj, history in self.confidence_history.items():
        #    print(f"{obj}: {history}")

        self.determine_ambience()

    def determine_ambience(self):
        weather = self.weather_counts.most_common(1)[0][0]
        place = self.place_counts.most_common(1)[0][0]
        scene = self.scene_counts.most_common(1)[0][0]
        time = self.time_counts.most_common(1)[0][0]

        if weather == "windy":
            self.ambience = "windy"
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
        else:
            self.ambience = "ambience"
    """
    def user_select_object(self):
        top_5 = self.get_top_5_objects()
        print("Top 5 detected objects:")
        for i, (obj, count) in enumerate(top_5, 1):
            print(f"{i}. {obj}: {count} occurrences")
        
        while True:
            try:
                choice = int(input("Select the most appropriate object (1-5): ")) - 1
                if 0 <= choice < 5:
                    return top_5[choice][0]
                else:
                    print("Please enter a number between 1 and 5.")
            except ValueError:
                print("Please enter a valid number.")

    """


    def calculate_intervals(self):
        for obj in self.object_status.keys():
            self.confidence_history[obj] = [0] * self.window_size  # Initialize confidence history

        for frame_idx, (objects, scores) in enumerate(zip(self.frame_objects, self.frame_values)):
            for obj in self.object_status.keys():
                if obj in objects:
                    score = scores[objects.index(obj)]
                else:
                    score = 0
                
                self.confidence_history[obj].append(score)
                self.confidence_history[obj] = self.confidence_history[obj][-self.window_size:]
                
                avg_score = np.mean(self.confidence_history[obj])
                
                if avg_score > self.start_threshold:
                    if not self.object_status[obj]['in_interval']:
                        self.start_new_interval(obj, frame_idx)
                elif avg_score < self.end_threshold:
                    if self.object_status[obj]['in_interval']:
                        if all(s < self.end_threshold for s in self.confidence_history[obj][-self.end_consecutive_frames:]):
                            self.end_interval(obj, frame_idx)

        # Handle objects that are still being detected at the end of the video
        for obj, status in self.object_status.items():
            if status['in_interval']:
                self.end_interval(obj, self.total_frames - 1)

        # Filter out too short intervals and ensure the end frame does not exceed the total number of frames
        for obj in self.object_intervals.keys():
            self.object_intervals[obj] = [
                (start, min(end, self.total_frames - 1), min((min(end, self.total_frames - 1) - start) / self.fps, (self.total_frames - 1 - start) / self.fps), needs_fine_sync)
                for start, end, duration, needs_fine_sync in self.object_intervals[obj]
                if (min(end, self.total_frames - 1) - start) / self.fps >= self.min_interval_duration
            ]

    def start_new_interval(self, obj, frame_idx):
        # end other interval
        for other_obj, status in self.object_status.items():
            if other_obj != obj and status['in_interval']:
                self.end_interval(other_obj, frame_idx - 1)
        
        self.object_status[obj]['in_interval'] = True
        self.object_status[obj]['start_frame'] = frame_idx
        self.object_status[obj]['needs_fine_sync'] = self.needs_fine_sync(obj)

    def end_interval(self, obj, end_frame):
        start_frame = self.object_status[obj]['start_frame']
        duration = (end_frame - start_frame) / self.fps
        needs_fine_sync = self.object_status[obj]['needs_fine_sync']
        self.object_intervals[obj].append((start_frame, end_frame, duration, needs_fine_sync))
        self.object_status[obj]['in_interval'] = False

    def get_intervals(self):
        self.object_intervals = {k: v for k, v in self.object_intervals.items() if v}
        return self.object_intervals
    
    def get_ambience(self):
        return self.ambience
    
    def get_object_infos(self):
        return self.object_infos
