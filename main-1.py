"""
Video Assembly API — TikTok-style video generator
Takes a list of scenes (image_url + voiceover text), and produces
a single vertical (1080x1920) MP4 with:
  - Ken Burns zoom/pan effect on each image
  - Burned-in captions synced to the voiceover
  - Scenes stitched together into one continuous video

Endpoints:
  GET  /              -> health check
  POST /create-video   -> returns final MP4 (binary)
"""

import os
import uuid
import shutil
import asyncio
import subprocess
import requests
import edge_tts
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List

app = FastAPI(title="Video Assembly API")

WIDTH, HEIGHT = 1080, 1920
FPS = 30
VOICE = "en-US-GuyNeural"


class Scene(BaseModel):
    image_url: str
    voiceover: str


class VideoRequest(BaseModel):
    scenes: List[Scene]
    voice: str = VOICE


@app.get("/")
def health_check():
    return {"status": "ok", "service": "video-assembly-server"}


def run(cmd: list):
    """Run a shell command, raise with full output on failure."""
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\n{result.stderr}")
    return result


async def generate_audio(text: str, voice: str, out_path: str):
    communicate = edge_tts.Communicate(text=text, voice=voice)
    await communicate.save(out_path)


def get_audio_duration(path: str) -> float:
    result = run([
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", path,
    ])
    return float(result.stdout.strip())


def download_image(url: str, out_path: str):
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    with open(out_path, "wb") as f:
        f.write(resp.content)


def build_srt(text: str, duration: float, out_path: str, words_per_chunk: int = 4):
    """Split voiceover into small caption chunks evenly spread across duration."""
    words = text.split()
    if not words:
        words = [" "]
    chunks = [
        " ".join(words[i:i + words_per_chunk])
        for i in range(0, len(words), words_per_chunk)
    ]
    per_chunk = duration / len(chunks)

    def fmt(t):
        h = int(t // 3600)
        m = int((t % 3600) // 60)
        s = int(t % 60)
        ms = int((t - int(t)) * 1000)
        return f"{h:02}:{m:02}:{s:02},{ms:03}"

    with open(out_path, "w", encoding="utf-8") as f:
        for i, chunk in enumerate(chunks):
            start = i * per_chunk
            end = (i + 1) * per_chunk
            f.write(f"{i+1}\n{fmt(start)} --> {fmt(end)}\n{chunk}\n\n")


def build_scene_clip(image_path: str, audio_path: str, srt_path: str, duration: float, out_path: str, zoom_in: bool):
    """Create one scene clip: Ken Burns zoom + burned captions + audio."""
    total_frames = int(duration * FPS)
    zoom_expr = (
        f"zoom+0.0015" if zoom_in else f"zoom-0.0015"
    )
    zoompan = (
        f"scale=-2:{HEIGHT*2}:flags=lanczos,"
        f"zoompan=z='{zoom_expr}':d={total_frames}:s={WIDTH}x{HEIGHT}:fps={FPS}"
    )
    subtitle_style = (
        "FontName=Arial,FontSize=16,PrimaryColour=&H00FFFFFF,"
        "OutlineColour=&H00000000,BorderStyle=1,Outline=3,Shadow=0,"
        "Alignment=2,MarginV=120"
    )
    vf = f"{zoompan},subtitles={srt_path}:force_style='{subtitle_style}'"

    run([
        "ffmpeg", "-y",
        "-loop", "1", "-i", image_path,
        "-i", audio_path,
        "-vf", vf,
        "-c:v", "libx264", "-t", str(duration),
        "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "128k",
        "-shortest",
        out_path,
    ])


def concat_clips(clip_paths: list, out_path: str, workdir: str):
    list_file = os.path.join(workdir, "concat_list.txt")
    with open(list_file, "w") as f:
        for p in clip_paths:
            f.write(f"file '{os.path.abspath(p)}'\n")
    run([
        "ffmpeg", "-y", "-f", "concat", "-safe", "0",
        "-i", list_file, "-c", "copy", out_path,
    ])


@app.post("/create-video")
async def create_video(req: VideoRequest):
    if not req.scenes:
        raise HTTPException(status_code=400, detail="'scenes' cannot be empty")

    job_id = str(uuid.uuid4())
    workdir = f"/tmp/video_{job_id}"
    os.makedirs(workdir, exist_ok=True)

    try:
        clip_paths = []
        for i, scene in enumerate(req.scenes):
            image_path = os.path.join(workdir, f"img_{i}.jpg")
            audio_path = os.path.join(workdir, f"audio_{i}.mp3")
            srt_path = os.path.join(workdir, f"cap_{i}.srt")
            clip_path = os.path.join(workdir, f"clip_{i}.mp4")

            download_image(scene.image_url, image_path)
            await generate_audio(scene.voiceover, req.voice, audio_path)
            duration = get_audio_duration(audio_path)
            build_srt(scene.voiceover, duration, srt_path)
            build_scene_clip(
                image_path, audio_path, srt_path, duration, clip_path,
                zoom_in=(i % 2 == 0),
            )
            clip_paths.append(clip_path)

        final_path = os.path.join(workdir, "final.mp4")
        concat_clips(clip_paths, final_path, workdir)

        return FileResponse(
            final_path,
            media_type="video/mp4",
            filename="video.mp4",
            background=None,
        )

    except Exception as e:
        shutil.rmtree(workdir, ignore_errors=True)
        raise HTTPException(status_code=500, detail=f"Video generation failed: {str(e)}")
