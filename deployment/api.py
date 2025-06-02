import logging
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import os
import uuid
import subprocess
import torch
import torchaudio
import whisper
from datetime import datetime
from typing import Dict
import asyncio
from inference import predict_fn, model_fn

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Video Sentiment Analysis API",
    description="Multimodal sentiment and emotion analysis for video files",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def install_ffmpeg():
    logger.debug("Checking FFmpeg installation")
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True, check=True)
        logger.debug("FFmpeg found")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        logger.error("FFmpeg not found. Please ensure FFmpeg is installed and added to PATH.")
        return False

if not install_ffmpeg():
    raise RuntimeError("FFmpeg is required but could not be installed")

UPLOAD_DIR = "Uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

jobs: Dict[str, Dict] = {}
video_files: Dict[str, Dict] = {}
api_quotas: Dict[str, Dict] = {
    "default_user": {
        "userId": "default_user",
        "maxRequests": 10000,
        "requestsUsed": 0
    }
}

def load_model():
    logger.debug("Loading model")
    try:
        model_dict = model_fn("model_normalized")
        return model_dict
    except Exception as e:
        logger.error("Model loading failed: %s", e)
        raise

model_dict = load_model()

def get_video_duration(video_path: str) -> float:
    try:
        cmd = [
            "ffprobe",
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            video_path
        ]
        duration = float(subprocess.check_output(cmd, stderr=subprocess.STDOUT).decode().strip())
        logger.debug("Video duration: %ss", duration)
        return duration
    except Exception as e:
        logger.error("Failed to get video duration: %s", e)
        return 30.0

async def analyze_video_async(video_path: str, job_id: str, user_id: str):
    logger.debug("Analyzing video %s for job %s", video_path, job_id)
    try:
        duration = get_video_duration(video_path)
        delay_time = duration * 0.8

        quota = api_quotas.get(user_id)
        if not quota or quota["requestsUsed"] >= quota["maxRequests"]:
            logger.error("API quota exceeded for user %s", user_id)
            raise HTTPException(status_code=429, detail="API quota exceeded")

        quota["requestsUsed"] += 1
        logger.debug("Quota updated: %d/%d", quota["requestsUsed"], quota["maxRequests"])

        await asyncio.sleep(delay_time)
        analysis = predict_fn({"video_path": video_path}, model_dict)

        jobs[job_id] = {
            "id": job_id,
            "userId": user_id,
            "status": "completed",
            "results": analysis,
            "is_mock": False,
            "filename": os.path.basename(video_path),
            "filePath": video_path,
            "createdAt": datetime.now().isoformat(),
            "fileSize": os.path.getsize(video_path),
            "processing_time": round(delay_time, 2)
        }
        logger.debug("Job %s completed with %d utterances", job_id, len(analysis["utterances"]))
    except Exception as e:
        logger.error("Analysis failed for video %s: %s", video_path, e)
        jobs[job_id] = {
            "id": job_id,
            "userId": user_id,
            "status": "failed",
            "results": {"error": str(e)},
            "is_mock": False,
            "filename": os.path.basename(video_path),
            "filePath": video_path,
            "createdAt": datetime.now().isoformat(),
            "fileSize": os.path.getsize(video_path),
            "processing_time": round(delay_time, 2)
        }
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": model_dict is not None,
        "device": str(model_dict.get("device", "unknown")),
        "timestamp": datetime.now().isoformat()
    }

@app.post("/upload")
async def upload_video(file: UploadFile = File(...), user_id: str = "default_user"):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")
    
    if not file.filename.lower().endswith(('.mp4', '.mov', '.avi')):
        raise HTTPException(status_code=400, detail="Invalid file type. Only .mp4, .mov, .avi are supported")
    
    job_id = str(uuid.uuid4())
    file_path = os.path.join(UPLOAD_DIR, f"{job_id}_{file.filename}")
    
    try:
        with open(file_path, "wb") as f:
            content = await file.read()
            if not content:
                raise HTTPException(status_code=400, detail="Empty file")
            f.write(content)
        
        video_files[job_id] = {
            "id": job_id,
            "userId": user_id,
            "key": file_path,
            "analyzed": False,
            "createdAt": datetime.now().isoformat()
        }
        
        logger.debug("Video uploaded: %s", file_path)
        return {"job_id": job_id, "status": "uploaded"}
    except Exception as e:
        logger.error("Upload failed: %s", e)
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=500, detail=f"Failed to upload file: {str(e)}")

@app.post("/analyze/{job_id}")
async def start_analysis(job_id: str, background_tasks: BackgroundTasks, user_id: str = "default_user"):
    try:
        job = jobs.get(job_id)
        if not job:
            video_file = video_files.get(job_id)
            if not video_file:
                raise HTTPException(status_code=404, detail="Job or video not found")
            file_path = video_file["key"]
        else:
            file_path = job["filePath"]

        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="Video file not found")

        jobs[job_id] = {
            "id": job_id,
            "userId": user_id,
            "status": "processing",
            "filename": os.path.basename(file_path),
            "filePath": file_path,
            "createdAt": datetime.now().isoformat(),
            "fileSize": os.path.getsize(file_path)
        }
        
        background_tasks.add_task(analyze_video_async, file_path, job_id, user_id)
        logger.debug("Analysis started for job %s", job_id)
        return {"job_id": job_id, "status": "started"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to start analysis: %s", e)
        raise HTTPException(status_code=500, detail=f"Failed to start analysis: {str(e)}")

@app.get("/status/{job_id}")
async def get_job_status(job_id: str):
    job = jobs.get(job_id)
    if not job:
        logger.warning("Job %s not found", job_id)
        raise HTTPException(status_code=404, detail="Job not found")
    return {
        "job_id": job_id,
        "status": job["status"],
        "is_mock": job.get("is_mock", False),
        "filename": job["filename"],
        "created_at": job["createdAt"],
        "file_size": job["fileSize"]
    }

@app.get("/results/{job_id}")
async def get_results(job_id: str):
    try:
        job = jobs.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
            
        if job["status"] == "processing":
            raise HTTPException(status_code=202, detail="Analysis in progress")
            
        if job["status"] == "failed":
            raise HTTPException(status_code=500, detail=f"Analysis failed: {job['results'].get('error', 'Unknown error')}")

        if not job.get("results"):
            raise HTTPException(status_code=404, detail="Results not found")
            
        logger.debug("Returning results for job %s", job_id)
        return {
            "results": job["results"],
            "is_mock": job.get("is_mock", False),
            "processing_time": job.get("processing_time", 0)
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get results: %s", e)
        raise HTTPException(status_code=500, detail=f"Failed to get results: {str(e)}")