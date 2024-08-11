import pandas as pd
import yt_dlp
import os
import subprocess
from tqdm import tqdm
import argparse
import time
import random

#python download_vggsound.py --csv_path "/home/s5614279/Master Project/AutoSFX/vgg_test/vggsound.csv" --output_dir "/home/s5614279/Master Project/audiosetdl/audioset_data/" --num_samples 2000

def download_video(video_id, label, output_dir, max_retries=3):
    parts = video_id.split('_')
    youtube_id = '_'.join(parts[:-1])  # 合并除最后一部分外的所有部分作为 YouTube ID
    time_stamp = parts[-1].replace('.mp4', '')
    temp_output_path = os.path.join(output_dir, f"{youtube_id}_temp.mp4")
    final_output_path = os.path.join(output_dir, video_id)
    
    if os.path.exists(final_output_path):
        return True

    try:
        start_time = int(time_stamp) / 1000  # Convert milliseconds to seconds
        print(f"Processing video {video_id} with start time: {start_time:.3f} seconds")
    except ValueError:
        print(f"Invalid time stamp for video {video_id}: {time_stamp}")
        return False

    ydl_opts = {
        'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
        'outtmpl': temp_output_path,
    }

    for attempt in range(max_retries):
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([f'https://www.youtube.com/watch?v={youtube_id}'])
            
            if os.path.exists(temp_output_path):
                ffmpeg_command = [
                    'ffmpeg', '-y', '-i', temp_output_path, 
                    '-ss', f"{start_time:.3f}", '-t', '10', 
                    '-c:v', 'libx264', '-c:a', 'aac', '-strict', 'experimental',
                    '-avoid_negative_ts', 'make_zero', '-async', '1',
                    final_output_path
                ]
                result = subprocess.run(ffmpeg_command, capture_output=True, text=True)
                if result.returncode != 0:
                    raise subprocess.CalledProcessError(result.returncode, ffmpeg_command, result.stdout, result.stderr)
                
                os.remove(temp_output_path)
                
                duration, actual_start = get_video_info(final_output_path)
                if abs(duration - 10) > 0.1:
                    print(f"Warning: Video {video_id} duration is {duration:.2f} seconds")
                    ffmpeg_command = [
                        'ffmpeg', '-y', '-i', final_output_path, 
                        '-t', '10', 
                        '-c:v', 'libx264', '-c:a', 'aac', '-strict', 'experimental',
                        f"{final_output_path}.tmp"
                    ]
                    result = subprocess.run(ffmpeg_command, capture_output=True, text=True)
                    if result.returncode != 0:
                        raise subprocess.CalledProcessError(result.returncode, ffmpeg_command, result.stdout, result.stderr)
                    os.replace(f"{final_output_path}.tmp", final_output_path)
                
                if abs(actual_start) > 0.1:
                    print(f"Warning: Video {video_id} actual start time is {actual_start:.2f} seconds")
                
                return True
        except Exception as e:
            print(f"Error processing video {video_id} (Attempt {attempt + 1}/{max_retries}): {e}")
            if os.path.exists(temp_output_path):
                os.remove(temp_output_path)
            if os.path.exists(final_output_path):
                os.remove(final_output_path)
            time.sleep(random.uniform(1, 3))

    return False

def get_video_info(file_path):
    duration_command = ['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'default=noprint_wrappers=1:nokey=1', file_path]
    start_command = ['ffprobe', '-v', 'error', '-show_entries', 'format=start_time', '-of', 'default=noprint_wrappers=1:nokey=1', file_path]
    
    duration_result = subprocess.run(duration_command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    start_result = subprocess.run(start_command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    
    duration = float(duration_result.stdout)
    start_time = float(start_result.stdout)
    
    return duration, start_time

def download_vggsound_videos(csv_path, output_dir, num_samples=200):
    df = pd.read_csv(csv_path, header=None, names=['video_id', 'label'])
    selected_samples = df.sample(n=min(num_samples, len(df)), random_state=42)
    os.makedirs(output_dir, exist_ok=True)

    successfully_downloaded = 0
    downloaded_videos = []

    for _, row in tqdm(selected_samples.iterrows(), total=len(selected_samples), desc="Downloading videos"):
        video_id = row['video_id']
        label = row['label']

        try:
            if download_video(video_id, label, output_dir):
                successfully_downloaded += 1
                downloaded_videos.append(row)
        except Exception as e:
            print(f"Error processing video {video_id}: {e}")
            continue

    print(f"Successfully downloaded {successfully_downloaded} out of {len(selected_samples)} videos.")

    if downloaded_videos:
        output_csv_path = os.path.join(output_dir, 'downloaded_videos.csv')
        pd.DataFrame(downloaded_videos).to_csv(output_csv_path, index=False, header=False)
        print(f"Saved information of downloaded videos to {output_csv_path}")
    else:
        print("No videos were successfully downloaded.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download random videos from VGG-Sound dataset")
    parser.add_argument("--csv_path", type=str, required=True, help="Path to the VGG-Sound CSV file")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save downloaded videos")
    parser.add_argument("--num_samples", type=int, default=50, help="Number of videos to download (default: 50)")
    
    args = parser.parse_args()
    
    csv_path = args.csv_path.strip('"').strip("'")
    output_dir = args.output_dir.strip('"').strip("'")
    
    download_vggsound_videos(csv_path, output_dir, args.num_samples)