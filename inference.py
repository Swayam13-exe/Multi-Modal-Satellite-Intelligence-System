import torch
from torchvision import transforms
from PIL import Image
import os
import config
from models.fusion_model import FusionModel
from utils.feature_engineering import extract_meta_features

class FusionPredictor:
    def __init__(self, model_path=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = FusionModel(num_classes=config.NUM_CLASSES)
        
        if model_path is None:
            model_path = os.path.join(config.MODEL_DIR, "best_fusion_model.pth")
            
        if os.path.exists(model_path):
            state_dict = torch.load(model_path, map_location=self.device)
            # Backward compatibility check for tabular encoder
            tab_weight_key = 'tabular_encoder.network.0.weight'
            if tab_weight_key in state_dict and state_dict[tab_weight_key].shape[1] == 4:
                print("Old 4-feature model detected. Applying backward compatibility padding for NDVI.")
                old_weight = state_dict[tab_weight_key]
                new_weight = torch.zeros((old_weight.shape[0], 5), device=old_weight.device)
                new_weight[:, :4] = old_weight
                state_dict[tab_weight_key] = new_weight
            
            self.model.load_state_dict(state_dict)
            print("Successfully loaded trained weights.")
        else:
            print("Warning: No pre-trained weights found. Using random initialization.")
            
        self.model.to(self.device)
        self.model.eval()
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225])
        ])

    def predict(self, image: Image.Image, lat: float, lon: float, month: int):
        """
        Runs inference on a single instance.
        """
        # Preprocess Image
        raw_tensor = self.transform(image)
        img_tensor = raw_tensor.unsqueeze(0).to(self.device)
        
        # Preprocess Metadata
        meta_features = extract_meta_features(lat, lon, month, image_tensor=raw_tensor)
        meta_tensor = torch.tensor(meta_features, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        # Inference
        with torch.no_grad():
            class_logits, veg_score, risk_logits = self.model(img_tensor, meta_tensor)
            
            # Post-process
            class_probs = torch.softmax(class_logits, dim=1)
            pred_class_idx = torch.argmax(class_probs, dim=1).item()
            pred_class_name = config.EUROSAT_CLASSES[pred_class_idx]
            
            veg_score_val = veg_score.item()
            
            risk_prob = torch.sigmoid(risk_logits).item()
            is_high_risk = risk_prob > 0.5
            
        return {
            'Land Use Class': pred_class_name,
            'Confidence': class_probs[0][pred_class_idx].item(),
            'Vegetation Score': veg_score_val,
            'Risk Indicator': "High Risk" if is_high_risk else "Normal",
            'Risk Probability': risk_prob
        }
