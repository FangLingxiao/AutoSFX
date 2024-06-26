import torch
import pandas as pd
import numpy as np
from MMPT.mmpt.models import MMPTModel
from tqdm import tqdm
from scipy.signal import find_peaks

class VideoAudioSync:
    def __init__(self, model_path="projects/retri/videoclip/how2.yaml"):
        self.model, self.tokenizer, self.aligner = MMPTModel.from_pretrained(model_path)
        self.model.eval()

    def load_video(self, video_path): #非正式函数
        # 需要一个处理视频的脚本
        # Returns a tensor of shape [T, C, H, W], where T is the total number of frames
        return torch.randn(3000, 3, 224, 224)

    def process_video_window(self, video_window):
        with torch.no_grad():
            video_features = self.model.encode_vision(video_window)
        return video_features

    def process_text(self, text):
        caps, cmasks = self.aligner._build_text_seq(
            self.tokenizer(text, add_special_tokens=False)["input_ids"]
        )
        caps, cmasks = caps[None, :], cmasks[None, :]  # bsz=1
        with torch.no_grad():
            text_features = self.model.encode_text(caps, cmasks)
        return text_features

    def compute_similarity(self, video_feat, text_feat):
        return torch.nn.functional.cosine_similarity(video_feat, text_feat, dim=1)

    def process_labels(self, labels_path, label_column):
        labels_df = pd.read_csv(labels_path)
        label_features = []
        labels = []

        for label in tqdm(labels_df[label_column], desc="Processing sound labels"):
            label_features.append(self.process_text(label))
            labels.append(label)

        return labels, label_features

    def detect_action_intervals(self, video_frames, label_features, window_size=60, step_size=1, min_duration=15):
        all_window_scores = []
        frame_scores = []

        for start in tqdm(range(0, len(video_frames) - window_size + 1, step_size), desc="Processing video windows"):
            window = video_frames[start:start+window_size].unsqueeze(0)
            window_features = self.process_video_window(window)
            
            window_scores = [self.compute_similarity(window_features, lf).item() for lf in label_features]
            all_window_scores.append(window_scores)
            frame_scores.extend([window_scores] * step_size)

        frame_scores = frame_scores[:len(video_frames)]

        threshold = np.percentile(all_window_scores, 75)
        high_score_regions = np.where(np.array(all_window_scores) > threshold)[0]

        for region_start in tqdm(high_score_regions, desc="Detailed analysis of high-scoring areas"):
            start_frame = region_start * step_size
            end_frame = min(start_frame + window_size, len(video_frames))
            
            for frame in range(start_frame, end_frame):
                frame_feature = self.process_video_window(video_frames[frame:frame+1].unsqueeze(0))
                frame_scores[frame] = [self.compute_similarity(frame_feature, lf).item() for lf in label_features]

        action_intervals = {}
        for label_idx in range(len(label_features)):
            label_scores = [scores[label_idx] for scores in frame_scores]
            peaks, _ = find_peaks(label_scores, height=threshold, distance=window_size//2)
            
            intervals = []
            for peak in peaks:
                start = peak
                end = peak
                while start > 0 and label_scores[start-1] > threshold:
                    start -= 1
                while end < len(label_scores)-1 and label_scores[end+1] > threshold:
                    end += 1
                
                duration = end - start + 1
                if duration >= min_duration:
                    intervals.append((start, end, duration))
            
            action_intervals[label_idx] = intervals

        return action_intervals, frame_scores

    def analyze_video(self, video_path, labels_path, label_column):
        video_frames = self.load_video(video_path)
        labels, label_features = self.process_labels(labels_path, label_column)
        action_intervals, frame_scores = self.detect_action_intervals(video_frames, label_features)

        # Calculate overall score
        average_scores = np.mean(frame_scores, axis=0)
        weighted_scores = np.average(frame_scores, axis=0, weights=frame_scores)
        top_5_counts = np.sum(np.array(frame_scores) >= np.percentile(frame_scores, 95, axis=0), axis=0)

        top_5_indices = np.argsort(average_scores)[-5:][::-1]
        top_5_indices_weighted = np.argsort(weighted_scores)[-5:][::-1]
        top_5_indices_frequency = np.argsort(top_5_counts)[-5:][::-1]

        return labels, action_intervals, top_5_indices, top_5_indices_weighted, top_5_indices_frequency, average_scores, weighted_scores, top_5_counts

def main():
    matcher = VideoAudioSync()
    video_path = "path/to/your/video.mp4"
    labels_path = "path/to/your/labels.csv"
    label_column = "label_text"

    labels, action_intervals, top_5_indices, top_5_indices_weighted, top_5_indices_frequency, average_scores, weighted_scores, top_5_counts = matcher.analyze_video(video_path, labels_path, label_column)

    print("\n方法1 - 平均分数:")
    for i, idx in enumerate(top_5_indices, 1):
        print(f"{i}. {labels[idx]} (平均相似度: {average_scores[idx]:.4f})")

    print("\n方法2 - 加权平均分数:")
    for i, idx in enumerate(top_5_indices_weighted, 1):
        print(f"{i}. {labels[idx]} (加权平均相似度: {weighted_scores[idx]:.4f})")

    print("\n方法3 - 出现频率:")
    for i, idx in enumerate(top_5_indices_frequency, 1):
        print(f"{i}. {labels[idx]} (出现在top 5的次数: {top_5_counts[idx]})")

    print("\n动作间隔:")
    for label_idx, intervals in action_intervals.items():
        print(f"标签 '{labels[label_idx]}':")
        for start, end, duration in intervals:
            print(f"  开始帧: {start}, 结束帧: {end}, 持续时间: {duration} 帧")

    # 假设视频帧率为30fps
    fps = 30
    results_df = pd.DataFrame({
        "Label": [labels[i] for i in top_5_indices],
        "Average_Similarity": average_scores[top_5_indices],
        "Weighted_Similarity": weighted_scores[top_5_indices_weighted],
        "Top5_Frequency": top_5_counts[top_5_indices_frequency],
        "Action_Intervals": [
            [f"{start/fps:.2f}-{end/fps:.2f}s ({duration/fps:.2f}s)" for start, end, duration in action_intervals[i]]
            for i in top_5_indices
        ]
    })
    results_df.to_csv("video_audio_matching_results.csv", index=False)
    print("\n结果已保存到 video_audio_matching_results.csv")

if __name__ == "__main__":
    main()