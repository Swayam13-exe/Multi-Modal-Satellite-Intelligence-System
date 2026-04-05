import os
import torch
from torchvision import transforms
from PIL import Image
from typing import Union, List, Dict

import config
from models.fusion_model import FusionModel
from utils.feature_engineering import extract_meta_features


class FusionPredictor:
    """
    Inference wrapper for the Multi-Modal Fusion Model.

    Loads the best saved checkpoint and exposes a clean `predict()` API
    for single-image and batch inference.

    Example
    -------
    >>> predictor = FusionPredictor()
    >>> img = Image.open("demo/Forest_sample.jpg")
    >>> result = predictor.predict(img, lat=20.59, lon=78.96, month=5)
    >>> print(result)
    {
        'Land Use Class': 'Forest',
        'Confidence': 0.987,
        'Vegetation Score': 0.91,
        'Risk Indicator': 'Normal',
        'Risk Probability': 0.03
    }
    """

    # ImageNet normalisation — matches ResNet18 pretraining
    MEAN = [0.485, 0.456, 0.406]
    STD  = [0.229, 0.224, 0.225]

    def __init__(self, model_path: str = None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = FusionModel(num_classes=config.NUM_CLASSES)

        if model_path is None:
            model_path = os.path.join(config.MODEL_DIR, "best_fusion_model.pth")

        if os.path.exists(model_path):
            state_dict = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            print(f"[FusionPredictor] Loaded weights from: {model_path}")
        else:
            print(f"[FusionPredictor] Warning: No weights found at '{model_path}'. "
                  "Run `python train.py` first. Using random initialisation.")

        self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.MEAN, std=self.STD),
        ])

    # ── Public API ────────────────────────────────────────────────────────────

    def predict(
        self,
        image: Image.Image,
        lat: float,
        lon: float,
        month: int,
    ) -> Dict:
        """
        Run inference on a single PIL image with geographic/temporal metadata.

        Parameters
        ----------
        image : PIL.Image.Image   RGB satellite patch (any size — resized internally)
        lat   : float             Latitude  in decimal degrees  [-90, 90]
        lon   : float             Longitude in decimal degrees  [-180, 180]
        month : int               Acquisition month             [1, 12]

        Returns
        -------
        dict with keys:
            Land Use Class, Confidence, Vegetation Score,
            Risk Indicator, Risk Probability
        """
        raw_tensor = self.transform(image)
        img_tensor = raw_tensor.unsqueeze(0).to(self.device)

        meta_features = extract_meta_features(lat, lon, month, image_tensor=raw_tensor)
        meta_tensor = torch.tensor(meta_features, dtype=torch.float32).unsqueeze(0).to(self.device)

        with torch.no_grad():
            class_logits, veg_score, risk_logits = self.model(img_tensor, meta_tensor)

        class_probs    = torch.softmax(class_logits, dim=1)
        pred_idx       = torch.argmax(class_probs, dim=1).item()
        pred_class     = config.EUROSAT_CLASSES[pred_idx]
        veg_score_val  = float(torch.clamp(veg_score, 0.0, 1.0).item())
        risk_prob      = float(torch.sigmoid(risk_logits).item())

        return {
            'Land Use Class'  : pred_class,
            'Confidence'      : round(float(class_probs[0][pred_idx].item()), 4),
            'Vegetation Score': round(veg_score_val, 4),
            'Risk Indicator'  : "High Risk" if risk_prob > 0.5 else "Normal",
            'Risk Probability': round(risk_prob, 4),
        }

    def predict_batch(
        self,
        samples: List[Dict],
    ) -> List[Dict]:
        """
        Batch inference for efficiency.

        Parameters
        ----------
        samples : list of dicts, each with keys: image, lat, lon, month

        Returns
        -------
        list of result dicts (same format as predict())
        """
        imgs, metas = [], []
        for s in samples:
            raw = self.transform(s['image'])
            imgs.append(raw.unsqueeze(0))
            meta = extract_meta_features(s['lat'], s['lon'], s['month'], image_tensor=raw)
            metas.append(torch.tensor(meta, dtype=torch.float32).unsqueeze(0))

        img_batch  = torch.cat(imgs).to(self.device)
        meta_batch = torch.cat(metas).to(self.device)

        with torch.no_grad():
            class_logits, veg_scores, risk_logits = self.model(img_batch, meta_batch)

        class_probs = torch.softmax(class_logits, dim=1)
        pred_idxs   = torch.argmax(class_probs, dim=1)
        risk_probs  = torch.sigmoid(risk_logits)

        results = []
        for i in range(len(samples)):
            idx       = pred_idxs[i].item()
            risk_prob = float(risk_probs[i].item())
            veg_val   = float(torch.clamp(veg_scores[i], 0.0, 1.0).item())
            results.append({
                'Land Use Class'  : config.EUROSAT_CLASSES[idx],
                'Confidence'      : round(float(class_probs[i][idx].item()), 4),
                'Vegetation Score': round(veg_val, 4),
                'Risk Indicator'  : "High Risk" if risk_prob > 0.5 else "Normal",
                'Risk Probability': round(risk_prob, 4),
            })
        return results