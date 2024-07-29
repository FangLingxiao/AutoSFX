
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
    'footsteps walking running', 'frog', 'cow', 'brushing_teeth', 'car_horn',
    'crackling_fire', 'helicopter', 'drinking_sipping', 'rain', 'insects',
    'laughing', 'hen', 'engine', 'breathing', 'crying_baby', 'hand_saw', 'coughing',
    'glass_breaking', 'snoring', 'toilet_flush', 'pig', 'washing_machine',
    'clock_tick', 'sneezing', 'rooster', 'sea_waves', 'siren', 'cat',
    'door_wood_creaks', 'crickets'
]

class Classify:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.clip_model, self.clip_preprocess = self.load_clip_model()
        self.last_conv_layer = self.clip_model.visual.transformer.resblocks[-1]
        self.clip_model.visual.transformer.register_forward_hook(self.save_transformer_input)
        self.transformer_input = None

    def save_transformer_input(self, module, input, output):
        self.transformer_input = input


    def load_clip_model(self):
        clip_model, preprocess = clip.load("ViT-B/32", device=self.device)
        return clip_model, preprocess
    
    def preprocess_image(self, image):
        preprocess = transforms.Compose([
            transforms.ToTensor()
        ])
        image_tensor = preprocess(image)
        #normalized_image_tensor = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), 
        #                                               (0.26862954, 0.26130258, 0.27577711))(image_tensor)
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
        weather_classes = ["sunny", "windy", "thunderstorm", "rainy", "drizzle"]
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
        with torch.no_grad():
            image_features = self.clip_model.encode_image(image_tensor)
            text_features = self.clip_model.encode_text(text_inputs)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        values, indices = similarity[0].topk(5)
        object_names = [ESC_50_classes[idx] for idx in indices]
        values = [value.item() for value in values]
        return values, object_names
    
    # Use grad-CAM to get the heatmap    
    def get_grad_cam(self, image):
        image_tensor = self.preprocess_image(image)
        image_tensor = image_tensor.unsqueeze(0)
        image_tensor.requires_grad_()

        # get visual transformer output
        with torch.enable_grad():
            features = self.clip_model.encode_image(image_tensor)
            output = features.mean()
        
        self.clip_model.zero_grad()
        output.backward()

        #output = self.clip_model.encode_image(image_tensor)
        
        #self.clip_model.zero_grad()
        #output.backward(torch.ones_like(output))
        
        # activations = self.last_conv_layer.forward(image_tensor)

        # get the last output of Transformer block
        activations = self.last_conv_layer.imput[0]

        #gradients = self.last_conv_layer.weight.grad.data
        gradients = activations.grad
        pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])

        #activations = self.last_conv_layer(image_tensor).detach()
        for i in range(activations.size(1)):
            activations[:, i, :, :] *= pooled_gradients[i]

        heatmap = torch.mean(activations, dim=1).squeeze()
        heatmap = F.relu(heatmap)
        heatmap /= torch.max(heatmap)

        return heatmap.detach().cpu().numpy()
    
    def analyze_heatmap(self, heatmap, threshold=0.5):
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

