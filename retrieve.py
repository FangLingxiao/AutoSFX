import os
import pandas as pd
import librosa
import numpy as np
import sync

class AudioRetriever:
    def __init__(self, csv_path, audio_folder):
        self.audio_data = pd.read_csv(csv_path)
        self.audio_folder = audio_folder

    def get_audio_duration(self, audio_file):
        try:
            y, sr = librosa.load(audio_file, sr=None)
            
            # 使用 librosa.effects.split 来找到所有非静音部分
            non_silent_intervals = librosa.effects.split(y, top_db=20)
            
            if len(non_silent_intervals) == 0:
                print(f"No valid segments found in audio: {audio_file}")
                return 0  # 没有有效音频
            
            # 获取前半部分有效音频的时长
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
                for _, row in possible_audios.iterrows():
                    audio_file = os.path.join(self.audio_folder, row['filename'])
                    if not os.path.exists(audio_file):
                        print(f"Audio file not found: {audio_file}")
                        continue
                    audio_duration = self.get_audio_duration(audio_file)
                    if audio_duration >= duration:
                        matched_audios[obj].append({
                            'interval': (start_frame, end_frame),
                            'audio_file': audio_file,
                            'audio_duration': audio_duration
                        })
                        print(f"Matched audio file: {audio_file} for object: {obj}")
                    else:
                        print(f"Audio file: {audio_file} duration {audio_duration:.2f} is less than interval duration {duration:.2f}")

        return matched_audios

def retrieve_audio(video_path, csv_path, audio_folder):
    syncer = sync.ObjectIntervalSync(video_path)
    syncer.analyze_frames()
    syncer.calculate_intervals()
    intervals = syncer.get_intervals()
    
    audio_retriever = AudioRetriever(csv_path, audio_folder)
    matched_audios = audio_retriever.match_audio_files(intervals)
    
    return matched_audios