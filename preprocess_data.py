import torch
from torch.utils.data import Dataset
from transformers import CLIPProcessor
import pandas as pd
import cv2
import numpy as np

class VideoDataset(Dataset):
    def __init__(self, csv_file, video_dir, num_frames=5):
        self.data = pd.read_csv(csv_file)
        self.video_dir = video_dir
        self.num_frames = num_frames
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        video_path = f"{self.video_dir}/{self.data.iloc[idx]['video_filename']}"
        label = self.data.iloc[idx]['label']
        
        frames = self.extract_frames(video_path)
        inputs = self.processor(text=[label], images=frames, return_tensors="pt", padding=True)
        
        return {
            'pixel_values': inputs['pixel_values'].squeeze(0),
            'input_ids': inputs['input_ids'].squeeze(0),
            'attention_mask': inputs['attention_mask'].squeeze(0)
        }
    
    def extract_frames(self, video_path):
        cap = cv2.VideoCapture(video_path)
        frames = []
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        step = max(frame_count // self.num_frames, 1)
        for i in range(0, min(frame_count, self.num_frames * step), step):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
            if len(frames) == self.num_frames:
                break
        cap.release()
        
        # Ensure we always return exactly num_frames frames
        if len(frames) < self.num_frames:
            last_frame = frames[-1] if frames else np.zeros((224, 224, 3), dtype=np.uint8)
            frames.extend([last_frame] * (self.num_frames - len(frames)))
        elif len(frames) > self.num_frames:
            frames = frames[:self.num_frames]
        
        return frames

def custom_collate(batch):
    pixel_values = torch.stack([item['pixel_values'] for item in batch])
    
    # Find max length for input_ids and attention_mask
    max_len = max(item['input_ids'].size(0) for item in batch)
    
    # Pad input_ids and attention_mask
    input_ids = torch.full((len(batch), max_len), 0, dtype=torch.long)
    attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long)
    
    for i, item in enumerate(batch):
        input_ids_len = item['input_ids'].size(0)
        input_ids[i, :input_ids_len] = item['input_ids']
        attention_mask[i, :input_ids_len] = item['attention_mask']
    
    return {
        'pixel_values': pixel_values,
        'input_ids': input_ids,
        'attention_mask': attention_mask
    }