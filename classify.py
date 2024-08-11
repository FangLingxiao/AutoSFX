
import torch
import clip
from torchvision import transforms
import cv2
import numpy as np
from PIL import Image
from torch.nn import functional as F
import os

ESC_50_classes = [
    'dog', 'chirping_birds', 'vacuum_cleaner', 'thunderstorm', 'door_wood_knock',
    'can_opening', 'crow', 'clapping', 'fireworks', 'chainsaw', 'airplane',
    'mouse_click', 'pouring_water', 'train', 'sheep', 'water_drops',
    'church_bells', 'clock_alarm', 'keyboard_typing', 'wind',
    'walking running', 'frog', 'cow', 'brushing_teeth', 'car_horn',
    'crackling_fire', 'helicopter', 'drinking_sipping', 'rain', 'insects',
    'laughing', 'hen', 'engine', 'breathing', 'crying_baby', 'hand_saw', 'coughing',
    'glass_breaking', 'snoring', 'toilet_flush', 'pig', 'washing_machine',
    'clock_tick', 'sneezing', 'rooster', 'sea_waves', 'siren', 'cat',
    'door_wood_creaks', 'crickets', 'high_heel'
]

class Classify:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        self.clip_model, self.clip_preprocess = self.load_clip_model()
        print("CLIP model loaded sucessfully")
        self.clip_model.visual.transformer.resblocks[-1].register_forward_hook(self.save_activations)
        self.clip_model.visual.transformer.resblocks[-1].register_full_backward_hook(self.save_gradients)
        self.activations = None
        self.gradients = None

    def save_activations(self, module, input, output):
        self.activations = output

    def save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]


    def load_clip_model(self):
        clip_model, preprocess = clip.load("ViT-B/32", device=self.device)
        return clip_model, preprocess
    
    def preprocess_image(self, image):
        preprocess = transforms.Compose([
            transforms.ToTensor()
        ])
        image_tensor = preprocess(image)
        print("Image tensor shape:", image_tensor.shape)
        return image_tensor.unsqueeze(0).to(self.device)
    
    def classify_place(self, image_tensor):
        place_classes = ["natrue", "urban"]
        text_inputs = torch.cat([clip.tokenize(f"a photo of {c}") for c in place_classes]).to(self.device)
        with torch.no_grad():
            image_features = self.clip_model.encode_image(image_tensor)
            text_features = self.clip_model.encode_text(text_inputs)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        probs = similarity.cpu().numpy()
        return place_classes[probs.argmax()]
    
    def classify_scene(self, image_tensor):
        text_inputs = clip.tokenize(["a photo of indoors", "a photo of outdoors"]).to(self.device)
        with torch.no_grad():
            image_features = self.clip_model.encode_image(image_tensor)
            text_features = self.clip_model.encode_text(text_inputs)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        probs = similarity.cpu().numpy()
        return "indoors" if probs[0][0] > probs[0][1] else "outdoors"
    
    def classify_weather(self, image_tensor):
        weather_classes = ["sunny", "windy"]
        text_inputs = torch.cat([clip.tokenize(f"a photo of a {w} day") for w in weather_classes]).to(self.device)
        with torch.no_grad():
            image_features = self.clip_model.encode_image(image_tensor)
            text_features = self.clip_model.encode_text(text_inputs)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        probs = similarity.cpu().numpy()
        return weather_classes[probs.argmax()]
    
    def classify_time(self, image_tensor):
        time_classes = ["day", "night"]
        text_inputs = torch.cat([clip.tokenize(f"a photo taken in the {t}") for t in time_classes]).to(self.device)
        with torch.no_grad():
            image_features = self.clip_model.encode_image(image_tensor)
            text_features = self.clip_model.encode_text(text_inputs)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        probs = similarity.cpu().numpy()
        return time_classes[probs.argmax()]
    
    def recognize_objects(self, image):
        image_tensor= self.preprocess_image(image)
        text_inputs = torch.cat([clip.tokenize(f"a photo of {e}") for e in ESC_50_classes]).to(self.device)
        
        with torch.enable_grad():
            image_features = self.clip_model.encode_image(image_tensor)

        torch.set_grad_enabled(False)
        text_features = self.clip_model.encode_text(text_inputs)

        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        values, indices = similarity[0].topk(5)
        object_names = [ESC_50_classes[idx] for idx in indices]
        values = [value.item() for value in values]

        torch.set_grad_enabled(True)
        return values, object_names
    
    def get_grad_cam(self, image):
        image_tensor = self.preprocess_image(image)
        image_tensor.requires_grad_()

        self.activations = None
        self.gradients = None

        # get visual transformer output
        features = self.clip_model.encode_image(image_tensor)
        output = features.mean()
        
        self.clip_model.zero_grad()
        output.backward()

        if self.activations is None or self.gradients is None:
            raise ValueError("Failed to capture activations or gradients")
        
        print("Activations shape:", self.activations.shape)
        print("Gradients shape:", self.gradients.shape)

        weights = self.gradients.mean([0, 1])  # Changed from mean(2)
        heatmap = (weights.unsqueeze(0).unsqueeze(0) * self.activations).sum(2)

        heatmap = F.relu(heatmap)
        if torch.max(heatmap) > 0:
            heatmap /= torch.max(heatmap)
        else:
            print("Warning: max value of heatmap is 0")

        heatmap = heatmap.squeeze().detach().cpu().numpy()
        
        # Reshape heatmap to a square shape if possible
        heatmap_size = int(np.sqrt(heatmap.shape[0]))
        if heatmap_size * heatmap_size == heatmap.shape[0]:
            heatmap = heatmap.reshape(heatmap_size, heatmap_size)
        else:
            # If we can't reshape to a square, we'll keep it as a 1D array
            print(f"Warning: Unable to reshape heatmap to a square. Shape: {heatmap.shape}")
        
        print("Heatmap shape:", heatmap.shape)
        print("Heatmap min:", heatmap.min(), "max:", heatmap.max())

        return heatmap
    
    def analyze_heatmap(self, heatmap, threshold=0.5):
        # Reshape heatmap to 2D if it's 1D
        if len(heatmap.shape) == 1:
            heatmap = heatmap.reshape(-1, 1)
        
        # Normalize heatmap
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
        
        # Create binary map
        binary_map = (heatmap > threshold).astype(np.uint8)
        
        # Find contours
        contours, _ = cv2.findContours(binary_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        total_area = binary_map.shape[0] * binary_map.shape[1]
        object_info = []

        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 0:
                relative_area = area / total_area
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    relative_x = cx / binary_map.shape[1]
                    relative_y = cy / binary_map.shape[0]
                    object_info.append({
                        'area': relative_area,
                        'position': (relative_x, relative_y)
                    })

        return object_info
    
    def save_heatmap(self, heatmap, frame_number, output_dir, original_frame):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        if len(heatmap.shape) == 1:
            heatmap_size = int(np.ceil(np.sqrt(heatmap.shape[0])))
            heatmap = np.pad(heatmap, (0, heatmap_size**2 - heatmap.shape[0]), mode='constant')
            heatmap = heatmap.reshape(heatmap_size, heatmap_size)
        
        heatmap = heatmap.astype(np.float32)
        
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
        
        heatmap_resized = cv2.resize(heatmap, (original_frame.shape[1], original_frame.shape[0]), interpolation=cv2.INTER_LINEAR)
        
        heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
        
        if len(original_frame.shape) == 2:
            original_frame = cv2.cvtColor(original_frame, cv2.COLOR_GRAY2RGB)
        elif original_frame.shape[2] == 4:
            original_frame = cv2.cvtColor(original_frame, cv2.COLOR_RGBA2RGB)
        
        original_frame = original_frame.astype(np.uint8)
        
        superimposed_img = cv2.addWeighted(original_frame, 0.6, heatmap_color, 0.4, 0)
        
        cv2.imwrite(os.path.join(output_dir, f'heatmap_frame_{frame_number}.png'), superimposed_img)

        print(f"Saved heatmap for frame {frame_number}")
    
    def process_video(self, video_path, output_size=(224, 224), output_dir='heatmaps', save_interval=15):
        video_capture = cv2.VideoCapture(video_path)
        frames = []
        frame_number = 0

        while video_capture.isOpened():
            ret, frame = video_capture.read()
            if not ret:
                break

            resized_frame = cv2.resize(frame, output_size, interpolation=cv2.INTER_LINEAR)
            resized_frame = resized_frame.astype(np.uint8)
            frames.append(resized_frame)

            if frame_number % save_interval == 0:
                print(f"Processing frame {frame_number}")
                pil_image = Image.fromarray(cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB))
                heatmap = self.get_grad_cam(pil_image)
                self.save_heatmap(heatmap, frame_number, output_dir, frame) 

            frame_number += 1

        video_capture.release()
        print(f"Processed {frame_number} frames")
        return frames