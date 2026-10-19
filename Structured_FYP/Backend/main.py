import shutil
from pathlib import Path
from urllib.parse import quote

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from pipeline import run_punch_detection, get_models

BASE_URL = "http://127.0.0.1:8000"

UPLOAD_DIR = Path("temp_videos")
OUTPUT_DIR = Path("detected_punches")
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

app = FastAPI(title="Boxing Analytics Punch Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serves the exported punch clips
app.mount("/videos", StaticFiles(directory=str(OUTPUT_DIR)), name="videos")


@app.on_event("startup")
def load_models_on_startup():
    # Load the 3 Keras models once at startup instead of on every upload
    get_models()


@app.get("/")
def read_root():
    return {"message": "Punch Detection API is online and functional!"}


@app.post("/upload-video/")
def upload_video(file: UploadFile = File(...)):
    content_type = file.content_type or ""
    if not content_type.startswith("video/"):
        raise HTTPException(status_code=400, detail="Uploaded file is not a verified video standard.")


    safe_name = Path(file.filename).name
    input_video_path = UPLOAD_DIR / safe_name

    try:
        with input_video_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save incoming data: {str(e)}")
    finally:
        file.file.close()

    session_name = input_video_path.stem
    session_output_dir = OUTPUT_DIR / session_name
    session_output_dir.mkdir(exist_ok=True)

    try:
        results = run_punch_detection(
            video_path=str(input_video_path),
            output_dir=str(session_output_dir),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Core pipeline execution processing fault: {str(e)}")

    # Build the response, skipping 'NoPunch' clips
    punches = []
    punch_counts = {}

    for result in results:
        label = result["classification"]
        if label == "NoPunch":
            continue

        punch_counts[label] = punch_counts.get(label, 0) + 1

        video_url = f"{BASE_URL}/videos/{quote(session_name)}/{quote(result['clip_filename'])}"

        punches.append({
            "punch_id": result["punch_id"],
            "classification": label,
            "confidence": result["confidence"],
            "clip_filename": result["clip_filename"],
            "video_url": video_url,
            "feedback": result["feedback"],   # list of text strings for the app
        })

    return {
        "message": "Processing complete",
        "original_video": safe_name,
        "total_punches_found": len(punches),
        "punch_counts": punch_counts,
        "punches": punches,
    }
