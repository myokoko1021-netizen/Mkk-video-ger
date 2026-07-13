"""
Video Assembly API — TikTok-style video generator
Takes a list of scenes (image_url + voiceover text), and produces
a single vertical (1080x1920) MP4 with:
  - Ken Burns zoom/pan effect (6 rotating movement patterns)
  - Word-by-word karaoke-style burned captions, synced to the voiceover
  - Cross-fade transitions between scenes (with OOM-safe hard-cut fallback)
  - Synthesized ambient background music bed under the narration
    (fully generated with FFmpeg's built-in audio filters — no external
    audio files, so there is zero licensing cost or risk)

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
BGM_CACHE_PATH = "/tmp/bgm_pad_cache.mp3"
BGM_VOLUME = 0.12  # kept low so it never competes with the narration


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


async def generate_audio_with_timing(text: str, voice: str, out_path: str):
    """
    Synthesize narration audio and capture word-level timing (via edge-tts's
    streaming WordBoundary events) so captions can highlight word-by-word
    instead of static evenly-spaced chunks.

    Falls back to a plain (non-karaoke) save if the installed edge-tts
    version doesn't support streaming word boundaries for any reason -
    the pipeline should never hard-fail just because of this.
    """
    word_boundaries = []
    try:
        communicate = edge_tts.Communicate(text=text, voice=voice)
        with open(out_path, "wb") as f:
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    f.write(chunk["data"])
                elif chunk["type"] == "WordBoundary":
                    word_boundaries.append({
                        "text": chunk["text"],
                        "start": chunk["offset"] / 10_000_000,      # 100ns -> seconds
                        "duration": chunk["duration"] / 10_000_000,  # 100ns -> seconds
                    })
    except Exception as e:
        print(f"WordBoundary capture failed, falling back to plain TTS: {e}")
        word_boundaries = []
        communicate = edge_tts.Communicate(text=text, voice=voice)
        await communicate.save(out_path)
    return word_boundaries


def get_audio_duration(path: str) -> float:
    """Reads container duration via ffprobe - works for audio or video files."""
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


def _ass_time(t: float) -> str:
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = t % 60
    return f"{h:01}:{m:02}:{s:05.2f}"


def build_captions(text: str, duration: float, word_boundaries: list, out_path: str, words_per_line: int = 3):
    """
    Build a .ass subtitle file. If word-level timing is available, each line
    uses ASS karaoke tags (\\k) so words highlight one-by-one as they're
    spoken (the trending TikTok caption style). Otherwise falls back to
    evenly-spaced plain lines (still styled, just no per-word highlight).
    """
    header = (
        "[Script Info]\n"
        "ScriptType: v4.00+\n"
        f"PlayResX: {WIDTH}\n"
        f"PlayResY: {HEIGHT}\n"
        "WrapStyle: 2\n\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, "
        "Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
        "Alignment, MarginL, MarginR, MarginV, Encoding\n"
        # Primary = highlight color revealed as each word is "sung" (gold).
        # Secondary = base color before it's spoken (white).
        "Style: Default,Arial,58,&H0000D7FF,&H00FFFFFF,&H00000000,&H00000000,-1,0,0,0,100,100,0,0,1,3,0,2,60,60,140,1\n\n"
        "[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    )

    lines = []

    if word_boundaries:
        # Clip any word timing that overruns the actual audio duration.
        clean = [w for w in word_boundaries if w["start"] < duration]
        for i in range(0, len(clean), words_per_line):
            chunk = clean[i:i + words_per_line]
            start = chunk[0]["start"]
            end = min(chunk[-1]["start"] + chunk[-1]["duration"], duration)
            karaoke_text = "".join(
                f"{{\\k{max(int(w['duration'] * 100), 1)}}}{w['text']} " for w in chunk
            ).strip()
            lines.append(f"Dialogue: 0,{_ass_time(start)},{_ass_time(end)},Default,,0,0,0,,{karaoke_text}")
    else:
        # Fallback: evenly-spaced plain chunks (old behavior), no karaoke tags.
        words = text.split() or [" "]
        chunk_size = 4
        chunks = [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]
        per_chunk = duration / max(len(chunks), 1)
        for i, chunk in enumerate(chunks):
            start = i * per_chunk
            end = (i + 1) * per_chunk
            lines.append(f"Dialogue: 0,{_ass_time(start)},{_ass_time(end)},Default,,0,0,0,,{chunk}")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(header)
        f.write("\n".join(lines))


def build_scene_clip(image_path: str, audio_path: str, caption_path: str, duration: float, out_path: str, pattern_index: int):
    """
    Create one scene clip: Ken Burns-style movement + burned karaoke captions
    + narration audio (background music is mixed in later, once, over the
    final assembled video - not per-clip - to keep this step cheap).

    Rotates through 6 movement patterns (by pattern_index % 6):
      0 - Zoom In (center)
      1 - Zoom Out (center)
      2 - Pan Left -> Right (no zoom)
      3 - Pan Right -> Left (no zoom)
      4 - Zoom In + diagonal pan (corner -> center)
      5 - Static + subtle "breathing" zoom (minimal movement)
    """
    total_frames = max(int(duration * FPS), 1)
    scale_height = int(HEIGHT * 1.10)
    pattern = pattern_index % 6
    t = f"(on/{total_frames})"

    if pattern == 0:
        zoom_expr = f"1.0+0.10*{t}"
        x_expr = "iw/2-(iw/zoom/2)"
        y_expr = "ih/2-(ih/zoom/2)"
    elif pattern == 1:
        zoom_expr = f"1.10-0.10*{t}"
        x_expr = "iw/2-(iw/zoom/2)"
        y_expr = "ih/2-(ih/zoom/2)"
    elif pattern == 2:
        zoom_expr = "1.10"
        x_expr = f"(iw-iw/zoom)*{t}"
        y_expr = "ih/2-(ih/zoom/2)"
    elif pattern == 3:
        zoom_expr = "1.10"
        x_expr = f"(iw-iw/zoom)*(1-{t})"
        y_expr = "ih/2-(ih/zoom/2)"
    elif pattern == 4:
        zoom_expr = f"1.0+0.10*{t}"
        x_expr = f"(iw-iw/zoom)*(1-{t})"
        y_expr = f"(ih-ih/zoom)*(1-{t})"
    else:
        zoom_expr = f"1.0+0.025*{t}"
        x_expr = "iw/2-(iw/zoom/2)"
        y_expr = "ih/2-(ih/zoom/2)"

    zoompan = (
        f"scale=-2:{scale_height}:flags=fast_bilinear,"
        f"zoompan=z='{zoom_expr}':x='{x_expr}':y='{y_expr}':d={total_frames}:s={WIDTH}x{HEIGHT}:fps={FPS}"
    )
    vf = f"{zoompan},subtitles={caption_path}"

    run([
        "ffmpeg", "-y",
        "-loop", "1", "-i", image_path,
        "-i", audio_path,
        "-vf", vf,
        "-c:v", "libx264", "-preset", "ultrafast", "-threads", "2",
        # Bitrate cap so the final assembled video reliably stays under
        # TikTok's 64MB single-chunk upload limit (a video was measured at
        # 70.4MB before this cap - just over the 67,108,864 byte ceiling).
        "-b:v", "2000k", "-maxrate", "2200k", "-bufsize", "4400k",
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
        # Same bitrate cap as build_scene_clip - this pass re-encodes
        # everything again for the crossfade, so it needs its own cap too.
        "-b:v", "2000k", "-maxrate", "2200k", "-bufsize", "4400k",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "96k",
        out_path,
    ])


def ensure_bgm_pad(path: str = BGM_CACHE_PATH, seed_duration: float = 24.0) -> str:
    """
    Generate a short ambient pad loop once, cached on disk, using only
    FFmpeg's built-in audio synthesis (sine oscillators + tremolo +
    lowpass). Fully synthetic - no external audio file, so zero licensing
    risk and zero cost. Looped via -stream_loop when mixed into the final
    video, so a short ~24s seed is enough for any video length.
    """
    if os.path.exists(path):
        return path

    run([
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", f"sine=frequency=98:duration={seed_duration}",
        "-f", "lavfi", "-i", f"sine=frequency=147:duration={seed_duration}",
        "-f", "lavfi", "-i", f"sine=frequency=196:duration={seed_duration}",
        "-filter_complex",
        "[0:a]volume=0.22[v0];[1:a]volume=0.16[v1];[2:a]volume=0.12[v2];"
        "[v0][v1][v2]amix=inputs=3:duration=longest[mixed];"
        "[mixed]tremolo=f=0.2:d=0.4,lowpass=f=1400,volume=2.0[out]",
        "-map", "[out]",
        "-c:a", "libmp3lame", "-b:a", "128k",
        path,
    ])
    return path


def mix_background_music(video_path: str, duration: float, bgm_path: str, out_path: str, bgm_volume: float = BGM_VOLUME):
    """
    Overlay the ambient background music bed under the narration of the
    final assembled video. Video stream is stream-copied (-c:v copy, no
    re-encode) so this stays cheap on memory - only the audio is mixed.
    """
    run([
        "ffmpeg", "-y",
        "-i", video_path,
        "-stream_loop", "-1", "-i", bgm_path,
        "-filter_complex",
        f"[1:a]volume={bgm_volume}[bg];[0:a][bg]amix=inputs=2:duration=first:dropout_transition=0[aout]",
        "-map", "0:v", "-map", "[aout]",
        "-c:v", "copy",
        "-c:a", "aac", "-b:a", "128k",
        "-t", str(duration),
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
            caption_path = os.path.join(workdir, f"cap_{i}.ass")
            clip_path = os.path.join(workdir, f"clip_{i}.mp4")

            download_image(scene.image_url, image_path)
            word_boundaries = await generate_audio_with_timing(scene.voiceover, req.voice, audio_path)
            duration = get_audio_duration(audio_path)
            build_captions(scene.voiceover, duration, word_boundaries, caption_path)
            build_scene_clip(
                image_path, audio_path, caption_path, duration, clip_path,
                pattern_index=i,
            )
            clip_paths.append(clip_path)
            durations.append(duration)

        narration_path = os.path.join(workdir, "narration_only.mp4")

        # Try smooth cross-fade transitions first; if it fails for any reason
        # (most likely OOM - all clips are decoded/re-encoded in one pass),
        # fall back to a plain hard-cut concat so the request still succeeds.
        try:
            crossfade_concat(clip_paths, durations, narration_path)
        except Exception as xfade_err:
            print(f"Cross-fade failed, falling back to hard-cut concat: {xfade_err}")
            concat_clips(clip_paths, narration_path, workdir)

        final_path = os.path.join(workdir, "final.mp4")
        total_duration = get_audio_duration(narration_path)

        # Add the synthesized background music bed. This is a cheap pass
        # (video is stream-copied, only audio is re-encoded) - but still
        # wrapped defensively so a music-mixing failure never breaks the
        # whole request; we just ship the narration-only video instead.
        try:
            bgm_path = ensure_bgm_pad()
            mix_background_music(narration_path, total_duration, bgm_path, final_path)
        except Exception as bgm_err:
            print(f"Background music mix failed, shipping narration-only video: {bgm_err}")
            final_path = narration_path

        return FileResponse(
            final_path,
            media_type="video/mp4",
            filename="video.mp4",
            background=None,
        )

    except Exception as e:
        shutil.rmtree(workdir, ignore_errors=True)
        raise HTTPException(status_code=500, detail=f"Video generation failed: {str(e)}")
