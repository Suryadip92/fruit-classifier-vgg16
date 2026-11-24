# Fruit Classifier — VGG16 Transfer Learning

A fruit image classifier built with PyTorch (fine-tuned VGG16, 16 classes) and served through FastAPI. The same server hosts the REST API and the website on a single port.

## Features

- Upload, drag & drop, or capture a photo with the camera
- 16-class prediction with confidence scores and full probability breakdown
- Responsive UI for phone, tablet, and desktop
- Single-origin FastAPI app: frontend + API on port 8000
- Works through one ngrok tunnel

## Project structure

```
backend/    FastAPI app, model wrapper, requirements
frontend/   index.html, style.css, app.js
artifacts/  best_fruit_vgg16.pth (trained weights, stored via Git LFS)
dataset/    image/ with 16 class folders
notebook.ipynb
```

## Setup

```bash
pip install -r backend/requirements.txt
```

Model weights are tracked with Git LFS. If you cloned the repo without Git LFS:

```bash
git lfs install
git lfs pull
```

## Run

```bash
cd backend
python -m uvicorn main:app --host 127.0.0.1 --port 8000
```

Open http://localhost:8000 — API docs at http://localhost:8000/docs.

To share it publicly with a single tunnel:

```bash
ngrok http 8000
```

## API

| Method | Path | Description |
| --- | --- | --- |
| GET | `/health` | Model status and device |
| POST | `/predict` | Multipart image upload, returns class + probabilities |
| GET | `/status` | Service metadata |
| GET | `/docs` | Interactive API documentation |
