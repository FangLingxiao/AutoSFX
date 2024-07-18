import os
import pandas as pd
import librosa
import numpy as np
import sync
import heapq

class AudioRetriever:
    def __init__(self, csv_path, audio_folder):
        self.audio_data = pd.read_csv(csv_path)
        self.audio_folder = audio_folder

    def get_audio_duration(self, audio_file):
        try:
            y, sr = librosa.load(audio_file, sr=None)
            
            non_silent_intervals = librosa.effects.split(y, top_db=20)
            
            if len(non_silent_intervals) == 0:
                print(f"No valid segments found in audio: {audio_file}")
                return 0
            
            valid_duration = (non_silent_intervals[-1, 1] - non_silent_intervals[0, 0]) / sr
            
            print(f"Audio file: {audio_file}, Valid duration: {valid_duration:.2f} seconds")
            return valid_duration
        except Exception as e:
            print(f"Error processing audio file: {audio_file}, Error: {e}")
            return 0

    def match_audio_files(self, intervals):
        matched_audios = {}
        for obj, obj_intervals in intervals.items():
            matched_audios[obj] = []
            for start_frame, end_frame, duration in obj_intervals:
                possible_audios = self.audio_data[self.audio_data['category'] == obj]
                print(f"Object: {obj}, Interval duration: {duration:.2f} seconds")
                
                top_audios = []
                counter = 0
                
                for _, row in possible_audios.iterrows():
                    audio_file = os.path.join(self.audio_folder, row['filename'])
                    if not os.path.exists(audio_file):
                        print(f"Audio file not found: {audio_file}")
                        continue
                    
                    audio_duration = self.get_audio_duration(audio_file)
                    duration_diff = abs(audio_duration - duration)
                    
                    if audio_duration >= duration:
                        heapq.heappush(top_audios, (duration_diff, counter, {
                            'interval': (start_frame, end_frame),
                            'audio_file': audio_file,
                            'audio_duration': audio_duration
                        }))
                        counter += 1
                        if len(top_audios) > 5:
                            heapq.heappop(top_audios)
                
                matched_audios[obj].append([audio_info for _, _, audio_info in sorted(top_audios)])
                
                if top_audios:
                    print(f"Top 5 matched audio files for object: {obj}")
                    for i, (diff, _, audio) in enumerate(sorted(top_audios), 1):
                        print(f"{i}. {audio['audio_file']} (duration: {audio['audio_duration']:.2f}s, diff: {diff:.2f}s)")
                else:
                    print(f"No suitable audio found for object: {obj} with duration {duration:.2f}")
        
        return matched_audios

def retrieve_audio(video_path, csv_path, audio_folder):
    syncer = sync.ObjectIntervalSync(video_path)
    syncer.analyze_frames()
    syncer.calculate_intervals()
    intervals = syncer.get_intervals()
    
    audio_retriever = AudioRetriever(csv_path, audio_folder)
    matched_audios = audio_retriever.match_audio_files(intervals)
    
    return matched_audios