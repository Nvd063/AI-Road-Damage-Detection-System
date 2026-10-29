# AI-Powered Road Damage Detection & Reporting System

Deep Learning semester project using YOLO11s.

## Detection Classes
- Crack
- Pothole
- Surface Damage
- Manhole

## Model Performance (Independent Test)
- Precision: 44.14%
- Recall: 49.35%
- mAP50: 44.97%
- mAP50-95: 19.90%

## Project Structure
- frontend/: Next.js web application
- backend/: FastAPI inference service
- model/best.pt: Trained YOLO11s weights
- model/metadata.json: Model information
- notebooks/: Training notebook (to be added)
- dataset/: Dataset documentation (to be added)
- docs/: Report and presentation (to be added)

## Backend Setup
From the repository root:

```bash
pip install -r requirements.txt
uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

API documentation: http://localhost:8000/docs

## Frontend Setup
```bash
cd frontend
npm install
npm run dev
```

Set NEXT_PUBLIC_API_URL to the deployed backend URL.

## Evaluation
Weights: model/best.pt
Architecture: YOLO11s
Input size: 640
Framework: Ultralytics

See model/metadata.json for class mapping.

## Disclaimer
Academic prototype. Predictions require human verification.
