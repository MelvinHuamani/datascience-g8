Clasificación de enfermedades en hojas de papa

Aplicación de aprendizaje automático para clasificar imágenes de hojas de papa. El proyecto incluye una API para realizar predicciones y una interfaz web donde se puede subir una imagen y consultar las tres clases más probables.

Características

- Clasificación con una red neuronal ResNet18.
- Seis clases:
  - Bacterial Soft Rot
  - Fungal Late Blight
  - Healthy
  - Viral Leaf Roll
  - Viral PVX
  - Viral PVY
- Modelo entrenado durante 8 épocas, con una exactitud de validación reportada del 98,09 %.
- Inferencia en CPU.
- API desarrollada con FastAPI.
- Interfaz web desarrollada con Streamlit.
- Despliegue de la API y la interfaz como servicios separados en Render.

Tecnologías y bibliotecas

- Python
- PyTorch
- Torchvision
- FastAPI
- Streamlit
- Docker
- Render

Datos

Las imágenes utilizadas para entrenar y evaluar el modelo provienen del conjunto de datos [Potato Leaf Disease Dataset](https://data.mendeley.com/datasets/d5b3fzpw3g/1), publicado en Mendeley Data por Ayesha Banu y Kaushik Deb.

Estructura del proyecto

potato_mlops-main/
├── app/
│   └── streamlit_app.py
├── data/
├── docs/
├── models/
│   └── potato_resnet18.pth
├── src/
│   └── potato_mlops
├── .dockerignore
├── .gitignore
├── Dockerfile
├── pyproject.toml
├── requirements-web.txt
├── uv.lock
└── README.md


API
La API carga el modelo potato_resnet18.pth y permite consultar su estado o enviar una imagen para obtener una predicción.

API desplegada: https://potato-mlops.onrender.com

Interfaz web
La aplicación Streamlit permite subir una imagen y muestra las tres predicciones principales con sus porcentajes de confianza.

Interfaz desplegada: https://potato-mlops-8goa.onrender.com

Modelo
El modelo utilizado es una versión de ResNet18 entrenada para clasificar imágenes de hojas de papa. El archivo de pesos se encuentra en models/potato_resnet18.pth.
