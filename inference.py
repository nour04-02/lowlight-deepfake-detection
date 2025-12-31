import torch
from models import create_model
import os

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "..", "Models")


def load_model(model_name, weight_file):
    model = create_model(model_name)
    weight_path = os.path.join(MODELS_DIR, weight_file)

    state_dict = torch.load(weight_path, map_location=DEVICE)
    model.load_state_dict(state_dict)

    model.to(DEVICE)
    model.eval()
    return model


def predict(model, image_tensor):
    with torch.no_grad():
        output = model(image_tensor)
        prob = torch.softmax(output, dim=1)[0, 1].item()
        label = "fake" if prob > 0.5 else "real"
        return label, prob
