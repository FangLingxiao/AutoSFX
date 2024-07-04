import openai
import torch
import clip
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from ram.models import ram_plus
import cv2
import os

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
    "waiting_room", "wine_cellar""abbey", "airport", "amphitheater", "amusement_park", "aquarium",
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

class SceneUnderstanding:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=self.device)
        self.blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        self.blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(self.device)
        self.ram_model = ram_plus(pretrained='/home/s5614279/Master Project/ram_plus_swin_large_14m.pth',
                                  image_size=224,
                                  vit='swin_l').to(self.device)

        
    def classify_scene(self, image):
        image = self.clip_preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits_per_image, _ = self.clip_model(image, self.clip_preprocess(["indoors", "outdoors"]))
            probs = logits_per_image.softmax(dim=-1).cpu().numpy()
        return "indoors" if probs[0][0] > probs[0][1] else "outdoors"
    
    def classify_place(self, image):
        image = self.clip_preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            text_inputs = torch.cat([clip.tokenize(f"a photo of {c}") for c in places365_classes]).to(self.device)
            logits_per_image, logits_per_text = self.clip_model(image, text_inputs)
            probs = logits_per_image.softmax(dim=-1).cpu().numpy()
        return places365_classes[probs.argmax()]
    
    def classify_time(self, image):
        time_classes = ["morning", "afternoon", "evening", "night"]
        image = self.clip_preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            text_inputs = torch.cat([clip.tokenize(f"a photo taken in the {t}") for t in time_classes]).to(self.device)
            logits_per_image, logits_per_text = self.clip_model(image, text_inputs)
            probs = logits_per_image.softmax(dim=-1).cpu().numpy()
        return time_classes[probs.argmax()]

    def classify_weather(self, image):
        weather_classes = ["sunny", "foggy", "windy", "cloudy", "thunderstorm", "rainy", "drizzle", "snowy", "blizzard"]
        image = self.clip_preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            text_inputs = torch.cat([clip.tokenize(f"a photo of a {w} day") for w in weather_classes]).to(self.device)
            logits_per_image, logits_per_text = self.clip_model(image, text_inputs)
            probs = logits_per_image.softmax(dim=-1).cpu().numpy()
        return weather_classes[probs.argmax()]

    def generate_caption(self, image):
        inputs = self.blip_processor(images=image, return_tensors="pt").to(self.device)
        out = self.blip_model.generate(**inputs)
        caption = self.blip_processor.decode(out[0], skip_special_tokens=True)
        return caption

    def recognize_objects(self, image):
        result = self.ram_model(image)
        return result["tags"]

    def analyze_image(self, image):
        scene_type = self.classify_scene(image)
        location = self.classify_place(image)
        
        #if scene_type == "outdoors":
        time = self.classify_time(image)
        weather = self.classify_weather(image)
        
        objects = self.recognize_objects(image)
        caption = self.generate_caption(image)

        general_context = f"I see {objects}. I am {scene_type}. I am at {location}. The time is {time}. The weather is {weather}. Overall, I see {caption}."
        
        return {
            "objects": objects,
            "ambience":scene_type,
            "location": location,
            "time": time,
            "weather": weather,
            "caption": caption
        }, general_context
    
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

    
def ask_chatgpt(prompt):
    # 从环境变量中获取API密钥
    openai_api_key = os.getenv("OPENAI_API_KEY")
    
    # 设置API密钥
    openai.api_key = openai_api_key
    
    # 调用OpenAI API
    response = openai.Completion.create(
        engine="text-davinci-003",  # 使用的模型引擎
        prompt=prompt,              # 你的提示文本
        max_tokens=150              # 最大生成的token数量
    )
    
    # 返回生成的文本
    return response.choices[0].text.strip()

