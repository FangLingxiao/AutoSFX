import openai
import torch
import clip
from torchvision import transforms
from transformers import BlipProcessor, BlipForConditionalGeneration
import cv2
import os
from torchvision.datasets import CIFAR100
from openai import OpenAI


# Prepare the dataset
#cifar100 = CIFAR100(root=os.path.expanduser("~/.cache"), download=True, train=False)
places365_classes = [
    "airport_terminal", "art_gallery", "auditorium", "bakery", "bar",
    "bathroom", "bedroom", "bookstore", "bowling_alley", "buffet",
    "cafeteria", "candy_store", "casino", "children_room", "church_inside",
    "classroom", "cloister", "closet", "clothing_store", "concert_hall",
    "conference_room", "conference_center", "dining_room", "deli", "dentists_office",
    "doorway", "dorm_room", "elevator_lobby", "fastfood_restaurant", "fire_station",
    "food_court", "galley", "game_room", "garage_indoor", "gym",
    "hairsalon", "hospital_room", "hotel_room", "ice_cream_parlor", "indoor_gym",
    "indoor_market", "indoor_stadium", "inn", "jewelry_shop", "karaoke_room",
    "kitchen", "laundromat", "lecture_room", "library", "living_room",
    "lobby", "locker_room", "mall", "museum", "music_studio",
    "nursery", "office", "operating_room", "pantry", "pavilion",
    "pharmacy", "pizza_place", "playroom", "reception", "restaurant",
    "room", "sauna", "shoe_shop", "shopping_mall", "shower",
    "spa", "staircase", "storage_room", "supermarket", "swimming_pool",
    "synagogue", "teenager_room", "television_studio", "theater", "train_station",
    "waiting_room", "wine_cellar","abbey", "airport", "amphitheater", "amusement_park", "aquarium",
    "arch", "assembly_line", "athletic_field", "badlands", "barn",
    "baseball_field", "basketball_court", "beach", "boathouse", "botanical_garden",
    "bridge", "building_facade", "burial_site", "bus_station", "campsite",
    "campus", "canal", "castle", "cathedral", "cemetery",
    "chalet", "coast", "construction_site", "corn_field", "courtyard",
    "creek", "desert", "doorway", "driveway", "fairway",
    "farm", "field", "fishing_pier", "forest", "gas_station",
    "golf_course", "grape_field", "gravel_pit", "greenhouse", "harbor",
    "highway", "home", "hospital", "hotel", "house",
    "iceberg", "industrial_area", "island", "jail", "japanese_garden",
    "kennel", "lake", "landfill", "landscape", "lighthouse",
    "marsh", "meadow", "military_base", "mountain", "mountain_path",
    "natural_history_museum", "park", "parking_garage", "parking_lot", "pasture",
    "patio", "pavilion", "pier", "playground", "plaza",
    "promenade", "quarry", "railroad", "rainforest", "reservoir",
    "restaurant", "river", "rock_arch", "ruin", "school",
    "ski_resort", "skyscraper", "stadium", "stage", "stream",
    "street", "subway_station", "supermarket", "swimming_pool", "temple",
    "tent", "tower", "trailer_park", "train_station", "tundra",
    "underwater", "valley", "viaduct", "volcano", "waterfall",
    "wheat_field", "wind_farm", "yard", "zoo"
    ]
ESC_50_classes = [
    "Dog", "Rain", "Crying baby", "Door knock", "Helicopter","Horse",
    "Rooster", "Sea waves", "Sneezing", "Mouse click", "Chainsaw",
    "Pig", "Crackling fire", "Clapping", "Keyboard typing", "Siren",
    "Cow", "Crickets", "Breathing", "Door, wood creaks", "Car horn",
    "Frog", "Chirping birds", "Coughing", "Can opening", "Engine",
    "Cat", "Water drops", "Footsteps", "Washing machine", "Train","running",
    "Hen", "Wind", "Laughing", "Vacuum cleaner", "Church bells",
    "Insects (flying)", "Pouring water", "Brushing teeth", "Clock alarm", "Airplane",
    "Sheep", "Toilet flush", "Snoring", "Clock tick", "Fireworks",
    "Crow", "Thunderstorm", "Drinking, sipping", "Glass breaking", "Hand saw"
]

class SceneUnderstanding:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.clip_model, self.clip_preprocess = self.load_clip_model()
        self.blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        self.blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(self.device)

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
    
    def classify_place(self, image_tensor):
        text_inputs = torch.cat([clip.tokenize(f"a photo of {c}") for c in places365_classes]).to(self.device)
        with torch.no_grad():
            image_features = self.clip_model.encode_image(image_tensor)
            text_features = self.clip_model.encode_text(text_inputs)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        probs = similarity.cpu().numpy()
        return places365_classes[probs.argmax()]
    
    def classify_time(self, image_tensor):
        time_classes = ["morning", "afternoon", "evening", "night"]
        text_inputs = torch.cat([clip.tokenize(f"a photo taken in the {t}") for t in time_classes]).to(self.device)
        with torch.no_grad():
            image_features = self.clip_model.encode_image(image_tensor)
            text_features = self.clip_model.encode_text(text_inputs)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        probs = similarity.cpu().numpy()
        return time_classes[probs.argmax()]

    def classify_weather(self, image_tensor):
        weather_classes = ["sunny", "foggy", "windy", "cloudy", "thunderstorm", "rainy", "drizzle", "snowy", "blizzard"]
        text_inputs = torch.cat([clip.tokenize(f"a photo of a {w} day") for w in weather_classes]).to(self.device)
        with torch.no_grad():
            image_features = self.clip_model.encode_image(image_tensor)
            text_features = self.clip_model.encode_text(text_inputs)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        probs = similarity.cpu().numpy()
        return weather_classes[probs.argmax()]

    def generate_caption(self, image_tensor):
        inputs = self.blip_processor(images=image_tensor, return_tensors="pt").to(self.device)
        out = self.blip_model.generate(**inputs)
        caption = self.blip_processor.decode(out[0], skip_special_tokens=True)
        return caption

    def recognize_objects(self, image_tensor):
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

    def analyze_image(self, image):
        image_tensor= self.preprocess_image(image)
        location = self.classify_place(image_tensor)
        scene_type = self.classify_scene(image_tensor)
        
        time = self.classify_time(image_tensor)
        weather = self.classify_weather(image_tensor)
        
        values,objects = self.recognize_objects(image_tensor)
        caption = self.generate_caption(image_tensor)

        general_context = f"I see {objects}. I am {scene_type}. I am at {location}. The time is {time}. The weather is {weather}. Overall, I see {caption}."
        """
        This part is for chatgpt
            return {
            "objects": objects,
            "values": values,
            "ambience":scene_type,
            "location": location,
            "time": time,
            "weather": weather,
            "caption": caption
        }, general_context
        """
        return values, objects

    
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



"""
def ask_chatgpt(prompt):
    # get API key from .env
    client = OpenAI(os.environ.get("OPENAI_API_KEY"))
    openai_api_key = os.getenv("OPENAI_API_KEY")
    
    openai.api_key = openai_api_key
    
    completion = client.chat.completions.create(
        model = "gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You will be provided with a context describing a scene, and your task is to give the sound suggestions that collectively form the audio landscape of the described scene"},
            {"role": "user", "content": prompt}
        ],
        max_tokens=150 
    )
    
    return response['choices'][0]['message']['content'].strip()
"""


