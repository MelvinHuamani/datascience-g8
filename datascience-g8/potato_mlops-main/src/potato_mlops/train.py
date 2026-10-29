from pathlib import Path

import torch
from torch import nn
from torch.optim import AdamW
from torchvision.models import ResNet18_Weights, resnet18

from potato_mlops.data import create_dataloaders


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "models" / "potato_resnet18.pth"
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
EPOCHS = 8


def get_device() -> torch.device:
    return DEVICE


def build_model(num_classes: int) -> nn.Module:
    model = resnet18(weights=ResNet18_Weights.DEFAULT)

    # Como línea base, entrenamos la capa final y congelamos las demás.
    for parameter in model.parameters():
        parameter.requires_grad = False

    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def evaluate(
    model: nn.Module,
    loader,
    criterion: nn.Module,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.inference_mode():
        for images, labels in loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            logits = model(images)
            loss = criterion(logits, labels)

            total_loss += loss.item() * labels.size(0)
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)

    return total_loss / total, correct / total


def train() -> None:
    train_loader, validation_loader, classes = create_dataloaders()
    model = build_model(num_classes=len(classes)).to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=1e-3,
    )

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    best_accuracy = -1.0

    print(f"Dispositivo: {DEVICE}")
    print(f"Clases: {classes}")
    print(f"Entrenamiento: {len(train_loader.dataset)} imágenes")
    print(f"Validación: {len(validation_loader.dataset)} imágenes")

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * labels.size(0)
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)

        train_loss = total_loss / total
        train_accuracy = correct / total
        validation_loss, validation_accuracy = evaluate(
            model, validation_loader, criterion
        )

        print(
            f"Época {epoch + 1}/{EPOCHS} | "
            f"pérdida entrenamiento: {train_loss:.4f} | "
            f"precisión entrenamiento: {train_accuracy:.2%} | "
            f"pérdida validación: {validation_loss:.4f} | "
            f"precisión validación: {validation_accuracy:.2%}"
        )

        if validation_accuracy > best_accuracy:
            best_accuracy = validation_accuracy
            checkpoint = {
                "model_state_dict": {
                    key: value.detach().cpu()
                    for key, value in model.state_dict().items()
                },
                "classes": classes,
                "image_size": 224,
                "mean": (0.485, 0.456, 0.406),
                "std": (0.229, 0.224, 0.225),
                "architecture": "resnet18",
            }
            torch.save(checkpoint, MODEL_PATH)
            print(f"Mejor modelo guardado en: {MODEL_PATH}")

    print(f"Mejor precisión de validación: {best_accuracy:.2%}")


if __name__ == "__main__":
    train()

