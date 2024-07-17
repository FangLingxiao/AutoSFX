import pandas as pd
import os
import sync

class AudioRetriever:
    def __init__(self, csv_path, audio_folder):
        self.audio_data = pd.read_csv(csv_path)
        self.audio_folder = audio_folder

    def match_audio_files(self, intervals):
        matched_audios = {}

        for obj, obj_intervals in intervals.items():
            matched_audios[obj] = []
            for start_frame, end_frame, duration in obj_intervals:
                possible_audios = self.audio_data[self.audio_data['category'] == obj]
                print(f"Object: {obj}, Interval duration: {duration:.2f} seconds")
                for _, row in possible_audios.iterrows():
                    audio_file = os.path.join(self.audio_folder, row['filename'])
                    if os.path.exists(audio_file):
                        matched_audios[obj].append({
                            'interval': (start_frame, end_frame),
                            'audio_file': audio_file,
                            'audio_duration': 5  # 假设每个音频文件的时长为5秒
                        })
                        print(f"Matched audio file: {audio_file} for object: {obj}")
                        break  # 假设每个interval只匹配一个音频文件

        return matched_audios
    
def retrieve_audio(video_path, csv_path, audio_folder):
    syncer = sync.ObjectIntervalSync(video_path)
    syncer.analyze_frames()
    syncer.calculate_intervals()
    intervals = syncer.get_intervals()

    print("Intervals:", intervals)

    # Initialize AudioRetriever and match audio file 
    retriever = AudioRetriever(csv_path, audio_folder)
    matched_audios = retriever.match_audio_files(intervals)

    return matched_audios