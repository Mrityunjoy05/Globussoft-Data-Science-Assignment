import sys
from pathlib import Path

# Add parent directory to sys.path for modular imports
parent_dir = Path(__file__).resolve().parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from task2.predict import predict

app = FastAPI(title="Face Authentication API", version="1.0.0")

ALLOWED_TYPES = {"image/jpeg", "image/jpg", "image/png", "image/webp"}


@app.get("/")
def Home():
    return {"status": "ok", "service": "Face Authentication"}


@app.post("/verify")
async def verify_faces(
    image1: UploadFile = File(..., description="First face image"),
    image2: UploadFile = File(..., description="Second face image"),
):
    if image1.content_type not in ALLOWED_TYPES or image2.content_type not in ALLOWED_TYPES:
        raise HTTPException(status_code=400, detail="Only JPEG, PNG, or WebP images are accepted.")

    image1_bytes = await image1.read()
    image2_bytes = await image2.read()

    result = predict(image1_bytes, image2_bytes)

    if result.get("error"):
        raise HTTPException(status_code=422, detail=result["error"])

    return JSONResponse(content=result)