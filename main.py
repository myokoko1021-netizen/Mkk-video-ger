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
FPS = 24
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
        # Negative returncode means the process was killed by a signal
        # (e.g. -9 = SIGKILL, commonly from an out-of-memory condition).
        if result.returncode < 0:
            raise RuntimeError(
                f"Command was killed by signal {-result.returncode} "
                f"(likely out-of-memory). Command: {' '.join(cmd)}\n"
                f"Last output: {result.stderr[-1000:]}"
            )
        raise RuntimeError(f"Command failed (exit {result.returncode}): {' '.join(cmd)}\n{result.stderr[-2000:]}")
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


def build_scene_clip(image_path: str, audio_path: str, srt_path: str, duration: float, out_path: str, pattern_index: int):
    """
    Create one scene clip: Ken Burns-style movement + burned captions + audio.

    Rotates through 6 movement patterns (by pattern_index % 6) so a video with
    several scenes doesn't feel like the same zoom repeated over and over:
      0 - Zoom In (center)
      1 - Zoom Out (center)
      2 - Pan Left -> Right (no zoom)
      3 - Pan Right -> Left (no zoom)
      4 - Zoom In + diagonal pan (corner -> center)
      5 - Static + subtle "breathing" zoom (minimal movement)

    Memory note: upscale margin kept at 1.10x (only slightly above the prior
    1.08x) so this stays within the 1GB RAM limit that caused OOM kills
    before. No crossfade / multi-image blending here on purpose - those cost
    much more memory and are a separate, riskier upgrade.
    """
    total_frames = max(int(duration * FPS), 1)
    scale_height = int(HEIGHT * 1.10)
    pattern = pattern_index % 6

    # t = normalized progress through the clip, 0.0 -> 1.0
    t = f"(on/{total_frames})"

    if pattern == 0:  # Zoom In (center)
        zoom_expr = f"1.0+0.10*{t}"
        x_expr = "iw/2-(iw/zoom/2)"
        y_expr = "ih/2-(ih/zoom/2)"
    elif pattern == 1:  # Zoom Out (center)
        zoom_expr = f"1.10-0.10*{t}"
        x_expr = "iw/2-(iw/zoom/2)"
        y_expr = "ih/2-(ih/zoom/2)"
    elif pattern == 2:  # Pan Left -> Right
        zoom_expr = "1.10"
        x_expr = f"(iw-iw/zoom)*{t}"
        y_expr = "ih/2-(ih/zoom/2)"
    elif pattern == 3:  # Pan Right -> Left
        zoom_expr = "1.10"
        x_expr = f"(iw-iw/zoom)*(1-{t})"
        y_expr = "ih/2-(ih/zoom/2)"
    elif pattern == 4:  # Zoom In + diagonal pan (corner -> center)
        zoom_expr = f"1.0+0.10*{t}"
        x_expr = f"(iw-iw/zoom)*(1-{t})"
        y_expr = f"(ih-ih/zoom)*(1-{t})"
    else:  # pattern == 5: Static + subtle breathing zoom
        zoom_expr = f"1.0+0.025*{t}"
        x_expr = "iw/2-(iw/zoom/2)"
        y_expr = "ih/2-(ih/zoom/2)"

    zoompan = (
        f"scale=-2:{scale_height}:flags=fast_bilinear,"
        f"zoompan=z='{zoom_expr}':x='{x_expr}':y='{y_expr}':d={total_frames}:s={WIDTH}x{HEIGHT}:fps={FPS}"
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
        "-c:v", "libx264", "-preset", "ultrafast", "-threads", "2",
        "-t", str(duration),
        "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "96k",
        "-shortest",
        out_path,
    ])


def concat_clips(clip_paths: list, out_path: str, workdir: str):
    """Simple hard-cut concat (stream copy, cheapest on memory)."""
    list_file = os.path.join(workdir, "concat_list.txt")
    with open(list_file, "w") as f:
        for p in clip_paths:
            f.write(f"file '{os.path.abspath(p)}'\n")
    run([
        "ffmpeg", "-y", "-f", "concat", "-safe", "0",
        "-i", list_file, "-c", "copy", out_path,
    ])


def crossfade_concat(clip_paths: list, durations: list, out_path: str, transition: float = 0.4):
    """
    Stitch clips together with a short cross-fade (video: xfade, audio:
    acrossfade) between each pair instead of a hard cut.

    This decodes/re-encodes every clip in a single ffmpeg pass (needed for
    xfade to work), which uses noticeably more memory than concat_clips.
    Caller is expected to catch RuntimeError and fall back to concat_clips
    on OOM.
    """
    n = len(clip_paths)
    if n == 1:
        shutil.copy(clip_paths[0], out_path)
        return

    inputs = []
    for p in clip_paths:
        inputs += ["-i", p]

    filter_parts = []
    prev_v, prev_a = "0:v", "0:a"
    cumulative = durations[0]
    for i in range(1, n):
        # Guard against a transition longer than either clip involved.
        safe_transition = max(0.1, min(transition, durations[i - 1] - 0.1, durations[i] - 0.1))
        offset = max(cumulative - safe_transition, 0)
        vout, aout = f"v{i}", f"a{i}"
        filter_parts.append(
            f"[{prev_v}][{i}:v]xfade=transition=fade:duration={safe_transition:.3f}:offset={offset:.3f}[{vout}]"
        )
        filter_parts.append(
            f"[{prev_a}][{i}:a]acrossfade=d={safe_transition:.3f}[{aout}]"
        )
        prev_v, prev_a = vout, aout
        cumulative = offset + durations[i]

    filter_complex = ";".join(filter_parts)

    run([
        "ffmpeg", "-y", *inputs,
        "-filter_complex", filter_complex,
        "-map", f"[{prev_v}]", "-map", f"[{prev_a}]",
        "-c:v", "libx264", "-preset", "ultrafast", "-threads", "2",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "96k",
        out_path,
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
        durations = []
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
                pattern_index=i,
            )
            clip_paths.append(clip_path)
            durations.append(duration)

        final_path = os.path.join(workdir, "final.mp4")

        # Try smooth cross-fade transitions first; if it fails for any reason
        # (most likely OOM - all clips are decoded/re-encoded in one pass),
        # fall back to a plain hard-cut concat so the request still succeeds.
        try:
            crossfade_concat(clip_paths, durations, final_path)
        except Exception as xfade_err:
            print(f"Cross-fade failed, falling back to hard-cut concat: {xfade_err}")
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
