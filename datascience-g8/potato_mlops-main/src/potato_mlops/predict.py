import argparse
from pathlib import Path

import torch
from PIL import Image
from torch import nn
from torchvision import transforms
from torchvision.models import resnet18


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "models" / "potato_resnet18.pth"
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load_predictor():
    checkpoint = torch.load(MODEL_PATH, map_location="cpu")
    classes = checkpoint["classes"]

    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(classes))
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(DEVICE)
    model.eval()

    image_transform = transforms.Compose([
        transforms.Resize((checkpoint["image_size"], checkpoint["image_size"])),
        transforms.ToTensor(),
        transforms.Normalize(checkpoint["mean"], checkpoint["std"]),
    ])

    return model, classes, image_transform


def predict_image(image_path: str | Path) -> tuple[str, float]:
    model, classes, image_transform = load_predictor()

    with Image.open(image_path) as image:
        image_tensor = image_transform(image.convert("RGB")).unsqueeze(0).to(DEVICE)

    with torch.inference_mode():
        probabilities = torch.softmax(model(image_tensor), dim=1)[0]

    index = int(probabilities.argmax().item())
    return classes[index], float(probabilities[index].item())


def main() -> None:
    parser = argparse.ArgumentParser(description="Clasifica una hoja de papa.")
    parser.add_argument("image", type=Path, help="Ruta de la imagen JPG")
    args = parser.parse_args()

    if not args.image.is_file():
        raise FileNotFoundError(f"No se encontró la imagen: {args.image}")

    label, confidence = predict_image(args.image)
    print(f"Predicción: {label}")
    print(f"Confianza del modelo: {confidence:.2%}")
    print(f"Dispositivo: {DEVICE}")


if __name__ == "__main__":
    main()