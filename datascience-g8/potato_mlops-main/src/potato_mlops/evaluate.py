from pathlib import Path

import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch import nn
from torchvision.models import resnet18

from potato_mlops.data import create_dataloaders


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "models" / "potato_resnet18.pth"
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def main() -> None:
    _, validation_loader, classes = create_dataloaders()

    checkpoint = torch.load(MODEL_PATH, map_location="cpu")
    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(classes))
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(DEVICE)
    model.eval()

    true_labels = []
    predictions = []

    with torch.inference_mode():
        for images, labels in validation_loader:
            images = images.to(DEVICE)
            logits = model(images)
            predictions.extend(logits.argmax(dim=1).cpu().tolist())
            true_labels.extend(labels.tolist())

    print(f"Dispositivo: {DEVICE}")
    print(classification_report(
        true_labels,
        predictions,
        labels=list(range(len(classes))),
        target_names=classes,
        zero_division=0,
    ))
    print("Matriz de confusión (filas = reales, columnas = predichas):")
    print(confusion_matrix(
        true_labels,
        predictions,
        labels=list(range(len(classes))),
    ))


if __name__ == "__main__":
    main()