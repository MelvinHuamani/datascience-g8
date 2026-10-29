from io import BytesIO
from typing import Annotated

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError

from potato_mlops.predict import DEVICE, load_predictor


MAX_UPLOAD_BYTES = 10 * 1024 * 1024

model, classes, image_transform = load_predictor()

app = FastAPI(title="Potato Leaf Disease Classifier")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "device": str(DEVICE),
        "classes": classes,
    }


@app.post("/predict")
async def predict(
    file: Annotated[UploadFile, File(description="Imagen de una hoja de papa")],
):
    image_bytes = await file.read(MAX_UPLOAD_BYTES + 1)

    if len(image_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="La imagen supera el límite de 10 MB.")

    try:
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
    except (UnidentifiedImageError, OSError):
        raise HTTPException(status_code=400, detail="El archivo no es una imagen válida.")

    image_tensor = image_transform(image).unsqueeze(0).to(DEVICE)

    with torch.inference_mode():
        probabilities = torch.softmax(model(image_tensor), dim=1)[0]

    top_indices = torch.topk(probabilities, k=min(3, len(classes))).indices.tolist()

    return {
        "filename": file.filename,
        "predicted_class": classes[int(probabilities.argmax().item())],
        "confidence": float(probabilities.max().item()),
        "top_3": [
            {
                "class": classes[index],
                "score": float(probabilities[index].item()),
            }
            for index in top_indices
        ],
        "device": str(DEVICE),
    }
