from fastapi import FastAPI, HTTPException, File, UploadFile
import dlib
import cv2
import numpy as np
import skops.io as sio
import os
import io
from PIL import Image

app = FastAPI()

# Global variables for models
detector = None
sp = None
facerec = None
svm_model = None

@app.on_event("startup")
async def load_models():
    global detector, sp, facerec, svm_model
    
    print("Loading BearID networks and SVM model warmly into memory...")
    
    models_dir = os.environ.get("MODELS_DIR", "/opt/bearid-models")
    detector_path = os.path.join(models_dir, "bear_detector_mmod.dat")
    sp_path = os.path.join(models_dir, "bear_landmark_68_model.dat")
    facerec_path = os.path.join(models_dir, "bear_embed_network.dat")
    
    svm_path = os.environ.get("SVM_MODEL_PATH", "/opt/models/model.skops")

    if not all(os.path.exists(p) for p in [detector_path, sp_path, facerec_path]):
        print(f"Warning: BearID Dlib models missing in {models_dir}")
    else:
        try:
            detector = dlib.cnn_face_detection_model_v1(detector_path)
            sp = dlib.shape_predictor(sp_path)
            facerec = dlib.face_recognition_model_v1(facerec_path)
            print("Dlib models loaded successfully.")
        except Exception as e:
            print(f"Error loading Dlib models: {e}")

    if not os.path.exists(svm_path):
        print(f"Warning: SVM model missing at {svm_path}")
    else:
        try:
            svm_model = sio.load(svm_path, trusted=True)
            print("SVM model loaded successfully.")
        except Exception as e:
            print(f"Error loading SVM model: {e}")

@app.post("/v1/models/bearid:predict")
async def predict(file: UploadFile = File(...)):
    if detector is None or sp is None or facerec is None or svm_model is None:
        raise HTTPException(status_code=503, detail="Models not loaded properly")

    try:
        contents = await file.read()
        pil_image = Image.open(io.BytesIO(contents)).convert("RGB")
        img = np.array(pil_image)
        # Convert RGB to BGR for dlib/opencv consistency if needed
        # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # Detect faces
        dets = detector(img, 1)
        if len(dets) == 0:
            return {"predictions": [], "message": "No bears detected"}

        predictions = []
        for d in dets:
            # Get landmarks
            shape = sp(img, d.rect)
            # Extract face chip
            face_chip = dlib.get_face_chip(img, shape)
            # Compute embedding
            face_descriptor = facerec.compute_face_descriptor(face_chip)
            embedding = np.array(face_descriptor).reshape(1, -1)

            # Predict using SVM
            pred_id = svm_model.predict(embedding)[0]
            
            # Try to get probabilities if available
            prob = {}
            if hasattr(svm_model, "predict_proba"):
                proba = svm_model.predict_proba(embedding)[0]
                pred_class_idx = np.where(svm_model.classes_ == pred_id)[0][0]
                prob = {"confidence": float(proba[pred_class_idx])}

            predictions.append({
                "box": {
                    "left": d.rect.left(),
                    "top": d.rect.top(),
                    "right": d.rect.right(),
                    "bottom": d.rect.bottom()
                },
                "predicted_id": str(pred_id),
                **prob
            })

        return {"predictions": predictions}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
