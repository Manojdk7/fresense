import os
from pathlib import Path
import torch
import torchvision.models as models
from torchvision import transforms
from PIL import Image

try:
    from transformers import pipeline
except Exception as e:
    pipeline = None
    print(f"Warning: transformers pipeline unavailable: {e}")

BASE_DIR = Path(__file__).resolve().parents[1]
CLIP_MODEL_DIR = BASE_DIR / "02_models" / "clip-vit-base-patch32"

class FoodImageModel:
    def __init__(self, model_path=None):
        print("Initializing ENSEMBLE Model: MobileNetV2 + CLIP...")
        
        # ==========================================
        # Component 1: Load MobileNetV2 (.pth)
        # ==========================================
        # Use provided model_path first, then Downloads folder, then fallbacks
        self.mobilenet_path = model_path or r"C:\Users\lenovo\Downloads\food_spoilage_model_best.pth"
        self.mobilenet_loaded = False
        self.clip_loaded = False
        self.classifier = None
        
        if not os.path.exists(self.mobilenet_path):
            fallback_paths = [
                "food_spoilage_model_best.pth",
                os.path.join(os.path.dirname(__file__), '..', '02_models', 'food_spoilage_model_best.pth')
            ]
            for path in fallback_paths:
                if os.path.exists(path):
                    self.mobilenet_path = path
                    break
                    
        try:
            if os.path.exists(self.mobilenet_path):
                print(f"Loading MobileNetV2 from {self.mobilenet_path}...")
                checkpoint = torch.load(self.mobilenet_path, map_location=torch.device('cpu'), weights_only=False)
                self.class_names = checkpoint['class_names']
                num_classes = checkpoint['num_classes']
                
                # Rebuild MobileNetV2 architecture
                self.model = models.mobilenet_v2(weights=None)
                self.model.classifier = torch.nn.Sequential(
                    torch.nn.Dropout(0.3),
                    torch.nn.Linear(1280, 512),
                    torch.nn.ReLU(),
                    torch.nn.Dropout(0.2),
                    torch.nn.Linear(512, num_classes)
                )
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.model.eval()
                
                # Preprocessing used during training
                self.transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
                self.mobilenet_loaded = True
                print("Success: MobileNetV2 loaded successfully!")
            else:
                print(f"Warning: MobileNetV2 not found at {self.mobilenet_path}. Will fallback to CLIP for spoilage.")
        except Exception as e:
            print(f"Error loading MobileNetV2: {e}")
            
        # ==========================================
        # Component 2: Load CLIP
        # ==========================================
        if pipeline is None:
            raise ImportError("CLIP is required, but transformers.pipeline is not available.")
        elif not CLIP_MODEL_DIR.exists():
            raise FileNotFoundError(
                f"CLIP is required, but local model folder was not found at {CLIP_MODEL_DIR}. "
                "Download openai/clip-vit-base-patch32 into 02_models/clip-vit-base-patch32."
            )
        else:
            try:
                print(f"Loading CLIP Model from {CLIP_MODEL_DIR}...")
                self.classifier = pipeline(
                    "zero-shot-image-classification",
                    model=str(CLIP_MODEL_DIR),
                    tokenizer=str(CLIP_MODEL_DIR),
                    image_processor=str(CLIP_MODEL_DIR),
                )
                self.clip_loaded = True
                print("Success: CLIP Model loaded successfully!")
            except Exception as e:
                raise RuntimeError(f"CLIP is required but failed to load from {CLIP_MODEL_DIR}: {e}") from e
            
        self.candidate_labels = [
            # Fruits
            "a fresh apple", "a rotten apple", 
            "a fresh banana", "a rotten banana", 
            "a fresh orange", "a rotten orange", 
            "a fresh pomegranate", "a rotten pomegranate", 
            "a fresh mango", "a rotten mango", 
            "a fresh tomato", "a rotten tomato",
            # Vegetables
            "a fresh cucumber", "a rotten cucumber", 
            "a fresh bitter gourd", "a rotten bitter gourd", 
            "a fresh carrot", "a rotten carrot", 
            "a fresh potato", "a rotten potato", 
            "a fresh capsicum", "a rotten capsicum",
            # Other Foods
            "fresh bread", "stale bread", 
            "fresh milk", "spoiled milk", 
            "fresh cheese", "moldy cheese", 
            "fresh eggs", "rotten eggs", 
            "fresh chicken", "spoiled chicken"
        ]
        # ====================================================================
        # Phase 2.2: OOD (Out-of-Distribution) / Non-Food Guardrail
        # ====================================================================
        self.food_validation_labels = [
            "a clear photo of edible food, grocery item, fruit, or vegetable",
            "a photo of fresh or spoiled food",
        ]
        self.non_food_categories = {
            "Vehicle / Transportation": "a photo of a motor vehicle, automobile, car, or bicycle",
            "Electronic Device": "a photo of an electronic device, computer, smartphone, or keyboard",
            "Furniture / Interior": "a photo of furniture, sofa, chair, table, or room architecture",
            "Human / Person": "a photo of a human face, person, skin, or human body part",
            "Animal / Pet": "a photo of an animal, dog, cat, bird, or wildlife",
            "Document / Text": "a photo of paper documents, printed text, or books",
            "Household Tool / Object": "a photo of a household tool, metal hardware, or plastic container",
        }
        self.all_validation_labels = self.food_validation_labels + list(self.non_food_categories.values())

    def predict_spoilage(self, image_path):
        """
        Ensemble prediction: MobileNetV2 for Spoilage Score + CLIP for Food Type.
        Includes Phase 2.2 Out-of-Distribution (OOD) Guardrail.
        Returns: tuple (spoilage_score: float, prediction_label: str)
        """
        # ============== ERROR HANDLING ==============
        if not os.path.exists(image_path):
            print(f"Error: Image file not found at {image_path}")
            return 0.0, "ERROR_FILE_NOT_FOUND"
            
        try:
            img = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return 0.0, "ERROR_INVALID_IMAGE"

        # Phase 2.2: Out-of-Distribution (OOD) / Non-Food Guardrail
        if self.clip_loaded:
            try:
                validation = self.classifier(img, candidate_labels=self.all_validation_labels)
                top_label = validation[0]['label']
                food_score = max((item['score'] for item in validation if item['label'] in self.food_validation_labels), default=0.0)

                non_food_matches = [
                    (cat_name, item['score'])
                    for cat_name, desc in self.non_food_categories.items()
                    for item in validation if item['label'] == desc
                ]
                top_non_food_name, top_non_food_score = max(non_food_matches, key=lambda x: x[1]) if non_food_matches else ("Non-food object", 0.0)

                # Rejection criteria: top label is non-food, or food confidence < 40%, or non-food confidence exceeds food confidence
                if top_label not in self.food_validation_labels or food_score < 0.40 or top_non_food_score > food_score:
                    print(f"[OOD GUARD] Rejected non-food: {top_non_food_name} ({top_non_food_score:.2f}) vs food ({food_score:.2f})")
                    return 0.0, f"INVALID_NON_FOOD:{top_non_food_name}"
            except Exception as e:
                print(f"Warning: food-image validation skipped because CLIP failed: {e}")

        # ============== ENSEMBLE LOGIC ==============
        spoilage_score = 0.0
        food_type = "UNKNOWN"
        
        # 1. Run MobileNetV2 for Spoilage Score
        if self.mobilenet_loaded:
            try:
                img_tensor = self.transform(img).unsqueeze(0)
                with torch.no_grad():
                    output = self.model(img_tensor)
                    probs = torch.nn.functional.softmax(output, dim=1)[0]
                    top_idx = torch.argmax(probs).item()
                
                # Calculate spoilage score by summing probabilities of 'rotten' classes
                spoilage_score = sum(probs[i].item() for i, name in enumerate(self.class_names) if 'rotten' in name.lower())
                food_type = self.class_names[top_idx]
            except Exception as e:
                print(f"Error: MobileNetV2 inference error: {e}")
                
        # 2. Run CLIP for Food Type Identification
        try:
            if not self.clip_loaded:
                raise RuntimeError("CLIP is not loaded")
            clip_results = self.classifier(img, candidate_labels=self.candidate_labels)
            
            # Extract just the food name from the top prediction
            top_label = clip_results[0]['label']
            words_to_remove = ["a ", "fresh ", "rotten ", "stale ", "spoiled ", "moldy "]
            food_name_str = top_label
            for w in words_to_remove:
                food_name_str = food_name_str.replace(w, "")
                
            food_type = food_name_str.strip().upper().replace(" ", "_")
            
            # Fallback: If MobileNet wasn't loaded, use CLIP's spoilage score
            if not self.mobilenet_loaded:
                 spoilage_keywords = ['rotten', 'spoiled', 'stale', 'moldy']
                 spoilage_score = sum([
                     r['score'] for r in clip_results 
                     if any(k in r['label'].lower() for k in spoilage_keywords)
                 ])
        except Exception as e:
            if self.clip_loaded:
                print(f"Error: CLIP inference error: {e}")

        # 3. Combine Results
        condition = "ROTTEN" if spoilage_score > 0.5 else "FRESH"
        prediction_label = f"{condition}_{food_type}"
        
        # Clamp score between 0.0 and 1.0
        spoilage_score = max(0.0, min(1.0, spoilage_score))
        
        return float(spoilage_score), prediction_label


# ============== TEST SCRIPT ==============
if __name__ == "__main__":
    print("=" * 60)
    print("🚀 ENSEMBLE System: MobileNetV2 + CLIP")
    print("=" * 60)
    
    model = FoodImageModel()
    
    test_image = "test.jpg"
    if os.path.exists(test_image):
        print(f"\n🖼️  Found test image: {test_image}")
        score, prediction = model.predict_spoilage(test_image)
        
        print("\n" + "=" * 60)
        print("📋 FINAL REPORT")
        print("=" * 60)
        print(f"Detected Item  : {prediction}")
        print(f"Spoilage Risk  : {score:.2%}")
        
        if score > 0.5:
            print("Status         : 🔴 SPOILED / NOT SAFE")
        else:
            print("Status         : 🟢 FRESH / SAFE")
        print("=" * 60)
    else:
        print(f"\n⚠️  Test image '{test_image}' not found.")
