import os
import cv2
import pandas as pd
import librosa
import numpy as np
import sync
import heapq

class AudioRetriever:
    def __init__(self, csv_path, effect_folder, ambience_folder):
        self.effect_data = pd.read_csv(csv_path)
        self.effect_folder = effect_folder
        self.ambience_folder = ambience_folder

    def get_effect_duration(self, effect_file):
        try:
            y, sr = librosa.load(effect_file, sr=None)
            
            non_silent_intervals = librosa.effects.split(y, top_db=20)
            
            if len(non_silent_intervals) == 0:
                print(f"No valid segments found in effect: {effect_file}")
                return 0
            
            valid_duration = (non_silent_intervals[-1, 1] - non_silent_intervals[0, 0]) / sr
            
            print(f"effect file: {effect_file}, Valid duration: {valid_duration:.2f} seconds")
            return valid_duration
        except Exception as e:
            print(f"Error processing effect file: {effect_file}, Error: {e}")
            return 0

    def match_effect_files(self, intervals, object_infos, video_path, video_fps):
        matched_effects = {}
        for obj, obj_intervals in intervals.items():
            matched_effects[obj] = []
            for idx, (start_frame, end_frame, duration, needs_fine_sync) in enumerate(obj_intervals):
                possible_effects = self.effect_data[self.effect_data['category'] == obj]
                print(f"Object: {obj}, Interval duration: {duration:.2f} seconds")
                
                top_effects = []
                counter = 0
                
                for _, row in possible_effects.iterrows():
                    effect_file = os.path.join(self.effect_folder, row['filename'])
                    if not os.path.exists(effect_file):
                        print(f"effect file not found: {effect_file}")
                        continue
                    
                    effect_duration = self.get_effect_duration(effect_file)
                    duration_diff = abs(effect_duration - duration)
                    
                    if effect_duration >= duration:
                        object_info = object_infos[start_frame // 2]
                        effect_info = {
                            'interval': (start_frame, end_frame),
                            'effect_file': effect_file,
                            'effect_duration': effect_duration,
                            'object_info': object_info,
                            'needs_fine_sync': needs_fine_sync
                        }
                        
                        if needs_fine_sync:
                            effect_info['fine_sync_start'] = self.fine_sync_effect(
                                video_path, (start_frame, end_frame), effect_file, video_fps
                            )
                        
                        heapq.heappush(top_effects, (duration_diff, counter, effect_info))
                        counter += 1
                        if len(top_effects) > 5:
                            heapq.heappop(top_effects)
                
                matched_effects[obj].append([effect_info for _, _, effect_info in sorted(top_effects)])
                
                if top_effects:
                    print(f"Top 5 matched effect files for object: {obj}")
                    for i, (diff, _, effect) in enumerate(sorted(top_effects), 1):
                        print(f"{i}. {effect['effect_file']} (duration: {effect['effect_duration']:.2f}s, diff: {diff:.2f}s)")
                        if effect['needs_fine_sync']:
                            print(f"   Fine sync start: {effect['fine_sync_start']:.3f}s")
                else:
                    print(f"No suitable effect found for object: {obj} with duration {duration:.2f}")

        print("Matched effects:")
        for obj, effects in matched_effects.items():
            print(f"Object: {obj}")
            for effect_list in effects:
                for effect in effect_list:
                    print(f"  Effect file: {effect['effect_file']}")
                    print(f"  Needs fine sync: {effect.get('needs_fine_sync', False)}")
                    print(f"  Fine sync start: {effect.get('fine_sync_start', 'N/A')}")
            print("---")
        
        return matched_effects
    
    def fine_sync_effect(self, video_path, interval, effect_file, video_fps):
        video = cv2.VideoCapture(video_path)
        if not video.isOpened():
            print(f"Error opening video file: {video_path}")
            return 0
        
        video_fps = video.get(cv2.CAP_PROP_FPS)
        if video_fps <= 0:
            print(f"Invalid video FPS ({video_fps}) for {video_path}, using default 30 FPS")
            video_fps = 30

        video.set(cv2.CAP_PROP_POS_FRAMES, interval[0])
        frames = []
        for _ in range(interval[1] - interval[0]):
            ret, frame = video.read()
            if ret:
                frames.append(frame)
        video.release()

        if not frames:
            print(f"No frames found in interval {interval}")
            return 0

        # Calculate the amount of motion in a video clip
        motion = []
        for i in range(1, len(frames)):
            prev_gray = cv2.cvtColor(frames[i-1], cv2.COLOR_BGR2GRAY)
            curr_gray = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
            flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            motion.append(np.sum(np.abs(flow)))

        # Find the frame with the most motion
        if motion:
            max_motion_frame = np.argmax(motion) + interval[0]
        else:
            print(f"No motion detected in interval {interval}")
            max_motion_frame = interval[0]

        y, sr = librosa.load(effect_file)

        # Find the peaks of the audio
        onset_frames = librosa.onset.onset_detect(y=y, sr=sr)
        if len(onset_frames) > 0:
            peak_sample = onset_frames[0] * sr // librosa.time_to_frames(1, sr=sr)
        else:
            print(f"No peaks detected in audio file: {effect_file}")
            peak_sample = 0

        #video_fps = video.get(cv2.CAP_PROP_FPS)
        #if video_fps == 0:
        #    print(f"Invalid video FPS for {video_path}")
        #    return 0
        
        audio_start_time = (max_motion_frame - interval[0]) / video_fps - peak_sample / sr

        print(f"Max motion frame: {max_motion_frame}, Interval start: {interval[0]}, Video FPS: {video_fps}")
        print(f"Peak sample: {peak_sample}, Sample rate: {sr}")
        print(f"Calculated fine sync start: {audio_start_time}")

        return max(0, audio_start_time)

    def retrieve_ambience(self, ambience_type):
        ambience_file = os.path.join(self.ambience_folder, f"{ambience_type}.mp3")
        if os.path.exists(ambience_file):
            return {'ambience_file': ambience_file}
        else:
            print(f"Ambience file not found: {ambience_file}")
            return None

def retrieve_audio(video_path, csv_path, effect_folder, ambience_folder):
    syncer = sync.ObjectIntervalSync(video_path)
    syncer.analyze_frames()
    syncer.calculate_intervals()
    intervals = syncer.get_intervals()
    object_infos = syncer.get_object_infos()
    ambience_type = syncer.get_ambience()
    
    audio_retriever = AudioRetriever(csv_path, effect_folder, ambience_folder)
    matched_effect_audios = audio_retriever.match_effect_files(intervals, object_infos, video_path, syncer.fps)
    matched_ambience_audio = audio_retriever.retrieve_ambience(ambience_type)
    
    return matched_effect_audios, matched_ambience_audio, ambience_type