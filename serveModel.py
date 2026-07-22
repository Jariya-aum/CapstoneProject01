# serve_model.py
"""
Simple Model Server for Corn Classification
Port: 5001 (ใช้แทน MLflow ที่มีปัญหา)
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import uvicorn
from typing import List

# ===== Load Model =====
print("🚀 Loading model...")
try:
    model = joblib.load('models/best_model_RandomForest.pkl')
    feature_names = joblib.load('models/feature_names.pkl')
    print(f"✅ Model loaded successfully!")
    print(f"   Features: {len(feature_names)}")
except FileNotFoundError as e:
    print(f"❌ Error: Model file not found!")
    print(f"   Please train model first: python train_model.py")
    exit(1)

# ===== FastAPI App =====
app = FastAPI(title="Corn Classification API")

# ===== Request/Response Models =====
class PredictionRequest(BaseModel):
    inputs: List[List[float]]

class PredictionResponse(BaseModel):
    predictions: List[int]
    probabilities: List[List[float]]

# ===== Endpoints =====

@app.get("/ping")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

@app.get("/info")
async def model_info():
    """Model information endpoint"""
    return {
        "model": "RandomForest",
        "type": str(type(model).__name__),
        "n_features": len(feature_names),
        "features": feature_names[:10]  # แสดง 10 ตัวแรก
    }

@app.post("/invocations", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Prediction endpoint"""
    try:
        # Convert input
        X = np.array(request.inputs)
        
        # Validate shape
        if X.shape[1] != len(feature_names):
            raise HTTPException(
                status_code=400,
                detail=f"Expected {len(feature_names)} features, got {X.shape[1]}"
            )
        
        # Predict
        predictions = model.predict(X).tolist()
        probabilities = model.predict_proba(X).tolist()
        
        return PredictionResponse(
            predictions=predictions,
            probabilities=probabilities
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ===== Run Server =====
if __name__ == "__main__":
    print("\n" + "="*60)
    print("🌽 Corn Classification Model Server")
    print("="*60)
    print(f"   Model: RandomForest")
    print(f"   Features: {len(feature_names)}")
    print(f"   Port: 5001")
    print("="*60)
    print("\n📍 Endpoints:")
    print("   GET  http://127.0.0.1:5001/ping")
    print("   GET  http://127.0.0.1:5001/info")
    print("   POST http://127.0.0.1:5001/invocations")
    print("\n🔥 Starting server...\n")
    
    uvicorn.run(app, host="127.0.0.1", port=5001)