import os
import cv2
import torch
import clip
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import recall_score
import seaborn as sns
import argparse

# CLIP模型加载
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-L/14", device=device)

def extract_frames(video_path, num_frames=5):
    frames = []
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    
    for i in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(frame)
            frame = preprocess(frame)
            frames.append(frame)
    
    cap.release()
    return torch.stack(frames)

def predict_label_with_ranks(video_path, audio_labels):
    frames = extract_frames(video_path)
    frames = frames.to(device)
    
    with torch.no_grad():
        image_features = model.encode_image(frames)
        text_features = model.encode_text(clip.tokenize(audio_labels).to(device))
    
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
    similarity = similarity.mean(dim=0)  # Average over frames
    
    ranked_indices = similarity.argsort(descending=True)
    ranked_labels = [audio_labels[i] for i in ranked_indices]
    return ranked_labels

def evaluate_clip_matching(csv_path, video_dir, num_samples=100):
    df = pd.read_csv(csv_path, header=None, names=['video_id', 'label'])
    
    if num_samples < len(df):
        df = df.sample(n=num_samples, random_state=42)
    
    all_labels = df['label'].unique().tolist()
    results = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        video_path = os.path.join(video_dir, row['video_id'])
        if not os.path.exists(video_path):
            print(f"Video not found: {video_path}")
            continue
        
        true_label = row['label']
        ranked_labels = predict_label_with_ranks(video_path, all_labels)
        rank = ranked_labels.index(true_label) + 1
        in_top_5 = true_label in ranked_labels[:5]
        
        results.append({
            'video_id': row['video_id'],
            'true_label': true_label,
            'rank': rank,
            'in_top_5': in_top_5
        })
    
    return pd.DataFrame(results)

def calculate_metrics(df):
    median_rank = np.median(df['rank'])
    recall_at_5 = df['in_top_5'].mean()
    return median_rank, recall_at_5

def plot_results(df, output_dir):
    median_rank, recall_at_5 = calculate_metrics(df)

    plt.figure(figsize=(10, 6))
    metrics = ['Median Rank', 'Recall@5']
    values = [median_rank, recall_at_5]
    plt.bar(metrics, values)
    plt.title('CLIP Video-Audio Matching Performance')
    plt.ylabel('Score')
    for i, v in enumerate(values):
        plt.text(i, v, f'{v:.2f}', ha='center', va='bottom')
    plt.savefig(os.path.join(output_dir, 'clip_performance.png'))
    plt.close()

    plt.figure(figsize=(10, 6))
    sns.histplot(df['rank'], bins=50, kde=True)
    plt.title('Distribution of Ranks')
    plt.xlabel('Rank')
    plt.ylabel('Frequency')
    plt.savefig(os.path.join(output_dir, 'rank_distribution.png'))
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Evaluate CLIP on video-audio matching task")
    parser.add_argument("--csv_path", type=str, required=True, help="Path to the test CSV file")
    parser.add_argument("--video_dir", type=str, required=True, help="Directory containing the video files")
    parser.add_argument("--num_samples", type=int, default=100, help="Number of samples to evaluate")
    parser.add_argument("--output_dir", type=str, default=".", help="Directory to save output files")
    
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    try:
        results_df = evaluate_clip_matching(args.csv_path, args.video_dir, args.num_samples)
        
        if results_df.empty:
            print("No results were obtained. Please check your input data and paths.")
            return

        results_df.to_csv(os.path.join(args.output_dir, 'evaluation_results.csv'), index=False)
        
        median_rank, recall_at_5 = calculate_metrics(results_df)
        print("Evaluation Results:")
        print(f"Median Rank: {median_rank:.2f}")
        print(f"Recall@5: {recall_at_5:.2f}")

        plot_results(results_df, args.output_dir)
        print(f"Results and plots have been saved to {args.output_dir}")
    except Exception as e:
        print(f"An error occurred during evaluation: {e}")

if __name__ == "__main__":
    main()