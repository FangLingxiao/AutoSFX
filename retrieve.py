import os
import pandas as pd
from pydub import AudioSegment
import numpy as np
import sync

class AudioRetriever:
    def __init__(self, csv_path, audio_folder):
        self.audio_data = pd.read_csv(csv_path)
        self.audio_folder = audio_folder

    def get_audio_duration(self, audio_file):
        audio = AudioSegment.from_file(audio_file)
        loudness = np.array(audio.dBFS_array)
        valid_segments = loudness[loudness > 0]  # 分贝超过0dB的部分
        valid_duration = len(valid_segments) / 1000.0  # 转换为秒
        return valid_duration

    def match_audio_files(self, intervals):
        matched_audios = {}
        
        for obj, intervals in intervals.items():
            matched_audios[obj] = []
            for start_frame, end_frame, duration in intervals:
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
                        break  # 假设每个interval只匹配一个音频文件
        
        return matched_audios

def retrieve_audio(video_path, csv_path, audio_folder):
    # 使用sync模块处理视频，获取物品时间间隔
    syncer = sync.ObjectIntervalSync(video_path)
    syncer.calculate_intervals()
    intervals = syncer.get_intervals()

    # 初始化AudioRetriever并匹配音频文件
    retriever = AudioRetriever(csv_path, audio_folder)
    matched_audios = retriever.match_audio_files(intervals)

    return matched_audios