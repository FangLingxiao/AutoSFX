
import torch
import clip
from torchvision import transforms
import cv2
import numpy as np
from PIL import Image
from torch.nn import functional as F

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
        self.clip_model, self.clip_preprocess = self.load_clip_model()
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
        weather_classes = ["sunny", "windy", "thunderstorm", "rainy"]
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
    
    # Use grad-CAM to get the heatmap    
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
       
        # activations = self.last_conv_layer.forward(image_tensor)

        # get the last output of Transformer block
        # activations = self.last_conv_layer.input[0]

        if self.activations is None or self.gradients is None:
            raise ValueError("Failed to capture activations or gradients")
        
        # print("Activations shape:", self.activations.shape)
        # print("Gradients shape:", self.gradients.shape)

        #gradients = self.last_conv_layer.weight.grad.data
        #gradients = activations.grad
        #pooled_gradients = torch.mean(gradients, dim=[0, 1])

        weights = self.gradients.mean(2)

        #activations = self.last_conv_layer(image_tensor).detach()
        #for i in range(activations.size(2)):
        # activations[:, :, i, :, :] *= pooled_gradients[i]
        heatmap = (weights.unsqueeze(-1) * self.activations).sum(2) # [50, 1]

        heatmap = F.relu(heatmap) # [50]
        heatmap /= torch.max(heatmap)

        heatmap = heatmap.detach().cpu().numpy()

        return heatmap
    
    def analyze_heatmap(self, heatmap, threshold=0.5):
        # heatmap is 2D numpy array
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
        binary_map = (heatmap > threshold).astype(np.uint8)
        
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
    
    def process_video(self, video_path, output_size=(224, 224)):
        video_capture = cv2.VideoCapture(video_path)
        frames = []

        while video_capture.isOpened():
            ret, frame = video_capture.read()
            if not ret:
                break
            
            # Resize every frame
            resized_frame = cv2.resize(frame, output_size)
            frames.append(resized_frame)

        video_capture.release()
        return frames

