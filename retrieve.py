import os
import pandas as pd
import librosa
import numpy as np
import sync

class AudioRetriever:
    def __init__(self, csv_path, audio_folder, ambience_folder):
        self.audio_data = pd.read_csv(csv_path)
        self.audio_folder = audio_folder
        self.ambience_folder = ambience_folder

    def get_audio_duration(self, audio_file):
        try:
            y, sr = librosa.load(audio_file, sr=None)
            
            # use librosa.effects.split to find non-silent part
            non_silent_intervals = librosa.effects.split(y, top_db=20)
            
            if len(non_silent_intervals) == 0:
                print(f"No valid segments found in audio: {audio_file}")
                return 0  # if no valid sound effects
            
            # get valid sound
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
                
                best_audio = None
                min_duration_diff = float('inf')
                
                for _, row in possible_audios.iterrows():
                    audio_file = os.path.join(self.audio_folder, row['filename'])
                    if not os.path.exists(audio_file):
                        print(f"Audio file not found: {audio_file}")
                        continue
                    
                    audio_duration = self.get_audio_duration(audio_file)
                    duration_diff = abs(audio_duration - duration)
                    
                    if audio_duration >= duration and duration_diff < min_duration_diff:
                        min_duration_diff = duration_diff
                        best_audio = {
                            'interval': (start_frame, end_frame),
                            'audio_file': audio_file,
                            'audio_duration': audio_duration
                        }
                
                if best_audio:
                    matched_audios[obj].append(best_audio)
                    print(f"Best matched audio file: {best_audio['audio_file']} for object: {obj}")
                else:
                    print(f"No suitable audio found for object: {obj} with duration {duration:.2f}")
        
        return matched_audios
    
    def retrieve_ambience(self, ambience_type):
        ambience_file = os.path.join(self.ambience_folder, f"{ambience_type}.wav")
        if os.path.exists(ambience_file):
            duration = self.get_audio_duration(ambience_file)
            return {'audio_file': ambience_file, 'audio_duration': duration}
        else:
            print(f"Ambience file not found: {ambience_file}")
            return None

def retrieve_audio(video_path, csv_path, audio_folder, ambience_folder):
    syncer = sync.ObjectIntervalSync(video_path)
    syncer.analyze_frames()
    syncer.calculate_intervals()
    intervals = syncer.get_intervals()
    ambience_type = syncer.get_ambience()
    
    audio_retriever = AudioRetriever(csv_path, audio_folder, ambience_folder)
    matched_audios = audio_retriever.match_audio_files(intervals)
    ambience_audio = audio_retriever.retrieve_ambience(ambience_type)
    
    return matched_audios, ambience_audio, ambience_type