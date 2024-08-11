import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor
import os
import pandas as pd
import cv2
import numpy as np

class SingleFrameVideoDataset(Dataset):
    def __init__(self, csv_file, video_dir):
        self.data = pd.read_csv(csv_file)
        self.video_dir = video_dir
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        video_path = os.path.join(self.video_dir, self.data.iloc[idx]['video_filename'])
        label = self.data.iloc[idx]['label']
        
        # Load single frame from video
        frame = self.load_frame(video_path)
        
        # Process image and text
        inputs = self.processor(text=[label], images=[frame], return_tensors="pt", padding=True)
        
        return {
            'pixel_values': inputs['pixel_values'].squeeze(0),
            'input_ids': inputs['input_ids'].squeeze(0),
            'attention_mask': inputs['attention_mask'].squeeze(0)
        }
    
    def load_frame(self, video_path):
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        cap.release()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (224, 224))  # Resize to fixed dimensions
            return frame
        else:
            raise ValueError(f"Could not load frame from {video_path}")

def collate_fn(batch):
    pixel_values = torch.stack([item['pixel_values'] for item in batch])
    input_ids = pad_sequence([item['input_ids'] for item in batch], batch_first=True, padding_value=0)
    attention_mask = pad_sequence([item['attention_mask'] for item in batch], batch_first=True, padding_value=0)
    
    return {
        'pixel_values': pixel_values,
        'input_ids': input_ids,
        'attention_mask': attention_mask
    }

class CLIPSingleFrameModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        
    def forward(self, pixel_values, input_ids, attention_mask):
        vision_outputs = self.clip.vision_model(pixel_values)
        image_embeds = vision_outputs.last_hidden_state[:, 0, :]  # Use CLS token
        
        text_outputs = self.clip.text_model(input_ids=input_ids, attention_mask=attention_mask)
        text_embeds = text_outputs.last_hidden_state[:, 0, :]  # Use CLS token
        
        image_embeds = self.clip.visual_projection(image_embeds)
        text_embeds = self.clip.text_projection(text_embeds)
        
        # Normalize features
        image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
        
        # Compute similarity
        logit_scale = self.clip.logit_scale.exp()
        logits = logit_scale * image_embeds @ text_embeds.t()
        
        return logits

def contrastive_loss(logits):
    labels = torch.arange(logits.size(0), device=logits.device)
    loss = torch.nn.functional.cross_entropy(logits, labels)
    return loss

def train_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    for batch in tqdm(dataloader):
        pixel_values = batch['pixel_values'].to(device)
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        
        optimizer.zero_grad()
        
        logits = model(pixel_values, input_ids, attention_mask)
        
        loss = contrastive_loss(logits)

        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)

def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in tqdm(dataloader):
            pixel_values = batch['pixel_values'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            logits = model(pixel_values, input_ids, attention_mask)
            
            loss = contrastive_loss(logits)

            total_loss += loss.item()
    
    return total_loss / len(dataloader)


def main():
    # Hyperparameters
    batch_size = 32
    num_epochs = 50
    learning_rate = 1e-4
    
    # Paths
    csv_path = '/home/s5614279/Master Project/audiosetdl/processed_subset.csv'
    video_dir = '/home/s5614279/Master Project/audiosetdl/audioset_data/'
    
    
    # Create full dataset
    full_dataset = SingleFrameVideoDataset(csv_path, video_dir)
    print(f"Full dataset size: {len(full_dataset)}")
    
    # Split dataset
    total_size = len(full_dataset)
    train_size = int(0.7 * total_size)
    val_size = int(0.15 * total_size)
    test_size = total_size - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size])
    
    # Create dataloaders with custom collate function
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True, collate_fn=collate_fn)
    
    # ... (其余代码保持不变)

if __name__ == "__main__":
    main()