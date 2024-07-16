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
        y, sr = librosa.load(audio_file)
        S = librosa.stft(y)
        S_db = librosa.amplitude_to_db(np.abs(S), ref=np.max)
        valid_segments = S_db > 0  # 分贝超过0dB的部分

        # 将valid_segments沿时间轴求和以找到开始和结束的时间点
        time_valid = np.sum(valid_segments, axis=0) > 0
        non_silent_indices = np.where(time_valid)[0]

        if len(non_silent_indices) == 0:
            return 0  # 没有有效音频
        
        start_index = non_silent_indices[0]
        end_index = non_silent_indices[-1]

        # 计算中间所有音频的时长
        valid_duration = (end_index - start_index + 1) * (1 / sr) * (len(S_db) / len(time_valid))

        return valid_duration


    def match_audio_files(self, intervals):
        matched_audios = {}
        
        for obj, obj_intervals in intervals.items():
            matched_audios[obj] = []
            for start_frame, end_frame, duration in obj_intervals:
                possible_audios = self.audio_data[self.audio_data['category'] == obj]
                for _, row in possible_audios.iterrows():
                    audio_file = os.path.join(self.audio_folder, row['filename'])
                    audio_duration = self.get_audio_duration(audio_file)
                    if audio_duration >= duration:
                        matched_audios[obj].append({
                            'interval': (start_frame, end_frame),
                            'audio_file': audio_file,
                            'audio_duration': audio_duration
                        })
                        break  # one audio file for one interval
        
        return matched_audios

def retrieve_audio(video_path, csv_path, audio_folder):
    syncer = sync.ObjectIntervalSync(video_path)
    syncer.calculate_intervals()
    intervals = syncer.get_intervals()

    # Initialize AudioRetriever and match audio file 
    retriever = AudioRetriever(csv_path, audio_folder)
    matched_audios = retriever.match_audio_files(intervals)

    return matched_audios