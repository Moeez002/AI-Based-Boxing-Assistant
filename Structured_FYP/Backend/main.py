import os
import uuid
import shutil
import traceback
from pathlib import Path
from urllib.parse import quote

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from pipeline import run_punch_detection, get_models

# Set BASE_URL in the notebook (os.environ["BASE_URL"] = public_url) before starting uvicorn.
BASE_URL = os.getenv("BASE_URL", "https://carpenter-sulk-subtract.ngrok-free.dev").rstrip("/")

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

task_store = {}


@app.on_event("startup")
def load_models_on_startup():
    get_models()


@app.get("/")
def read_root():
    return {"message": "Punch Detection API is online and functional!"}


def process_video_background(task_id: str, input_video_path: Path, session_output_dir: Path, session_name: str, safe_name: str):
    try:
        task_store[task_id]["status"] = "processing"

        results = run_punch_detection(
            video_path=str(input_video_path),
            output_dir=str(session_output_dir),
        )

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
                "feedback": result["feedback"],
            })

        task_store[task_id] = {
            "status": "completed",
            "original_video": safe_name,
            "total_punches_found": len(punches),
            "punch_counts": punch_counts,
            "punches": punches,
        }
    except Exception as e:
        # Print the full traceback (file + line) to the server log (Colab output)
        traceback.print_exc()
        task_store[task_id] = {
            "status": "failed",
            "error": str(e)
        }


@app.post("/upload-video/")
async def upload_video(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    content_type = file.content_type or ""
    if not content_type.startswith("video/"):
        raise HTTPException(status_code=400, detail="Uploaded file is not a verified video standard.")

    safe_name = Path(file.filename).name
    input_video_path = UPLOAD_DIR / f"{uuid.uuid4()}_{safe_name}"

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

    task_id = str(uuid.uuid4())
    task_store[task_id] = {"status": "queued"}
    print(f"New task: {task_id}", flush=True)

    background_tasks.add_task(
        process_video_background,
        task_id,
        input_video_path,
        session_output_dir,
        session_name,
        safe_name
    )

    return {
        "status": "queued",
        "task_id": task_id,
        "message": "Video uploaded successfully and is processing in the background."
    }


@app.get("/task-status/{task_id}")
async def get_task_status(task_id: str):
    if task_id not in task_store:
        raise HTTPException(status_code=404, detail="Task ID not found.")

    return task_store[task_id]
