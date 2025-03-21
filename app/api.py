from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
import shutil
import os
from app.enhancer.processor import enhance_image

app = FastAPI()

UPLOAD_DIR = "temp_uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.post("/enhance/")
async def enhance(file: UploadFile = File(...), scale: int = 4):
    input_path = os.path.join(UPLOAD_DIR, file.filename)
    output_path = os.path.join(UPLOAD_DIR, f"enhanced_{file.filename}")

    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    enhance_image(input_path, output_path, scale=scale)
    return FileResponse(output_path, media_type="image/jpeg", filename=f"enhanced_{file.filename}")
