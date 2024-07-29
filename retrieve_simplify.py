import os
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

    def match_effect_files(self, intervals, object_infos):
        matched_effects = {}
        for obj, obj_intervals in intervals.items():
            matched_effects[obj] = []
            for idx, (start_frame, end_frame, duration) in enumerate(obj_intervals):
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
                        heapq.heappush(top_effects, (duration_diff, counter, {
                            'interval': (start_frame, end_frame),
                            'effect_file': effect_file,
                            'effect_duration': effect_duration,
                            'object_info': object_info
                        }))
                        counter += 1
                        if len(top_effects) > 5:
                            heapq.heappop(top_effects)
                
                matched_effects[obj].append([effect_info for _, _, effect_info in sorted(top_effects)])
                
                if top_effects:
                    print(f"Top 5 matched effect files for object: {obj}")
                    for i, (diff, _, effect) in enumerate(sorted(top_effects), 1):
                        print(f"{i}. {effect['effect_file']} (duration: {effect['effect_duration']:.2f}s, diff: {diff:.2f}s)")
                else:
                    print(f"No suitable effect found for object: {obj} with duration {duration:.2f}")
        
        return matched_effects

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
    matched_effect_audios = audio_retriever.match_effect_files(intervals, object_infos)
    matched_ambience_audio = audio_retriever.retrieve_ambience(ambience_type)
    
    return matched_effect_audios, matched_ambience_audio, ambience_type