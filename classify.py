
import torch
import clip
from torchvision import transforms
import cv2

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
