import torch
import timm
import cv2
import numpy as np

class ViTEmbedder:
    def __init__(self, model_name="mobilevit_s", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.model = timm.create_model(
            model_name,
            pretrained=True,
            num_classes=0  # Remove classification head: output embeddings only
        ).to(self.device)
        self.model.eval()

        # Default input size for model (224x224)
        self.input_size = 224

    def preprocess(self, frame):
        resized = cv2.resize(frame, (self.input_size, self.input_size))
        img = resized / 255.0
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0)
        return torch.tensor(img, dtype=torch.float32).to(self.device)

    def get_embedding(self, frame):
        x = self.preprocess(frame)
        with torch.no_grad():
            emb = self.model(x)
        return emb.cpu().numpy().flatten()
