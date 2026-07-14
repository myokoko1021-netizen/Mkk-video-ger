"""
Video Assembly API — TikTok-style video generator (movie-style motion, $0 cost)

Pipeline per scene:
  1. Download TWO image variants (same prompt, different Pollinations seed)
  2. Render each as its own short Ken Burns movement clip (video only)
  3. Cross-dissolve ("morph") the two clips across the full scene duration
  4. Generate a soft ambient background pad (ffmpeg-synthesized, no external
     files) and mix it under the voiceover at low volume
  5. Burn in captions + a subtle film-grain pass, then mux the mixed audio
Scenes are then stitched together with cross-fade transitions (with a
low-memory hard-cut fallback if the host runs low on RAM).

Everything uses ffmpeg's built-in filters and the free Pollinations/Edge-TTS
services — no paid APIs are used anywhere in this pipeline.

Endpoints:
  GET  /              -> health check
  POST /create-video   -> returns final MP4 (binary)
"""

import os
import uuid
import random
import shutil
import subprocess
import requests
import edge_tts
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List

app = FastAPI(title="Video Assembly API")

WIDTH, HEIGHT = 1080, 1920
FPS = 20
VOICE = "en-US-GuyNeural"

# Root frequencies (Hz) for the ambient pad, rotated per scene for subtle
# variety — a small set of low, "space-y" chord roots.
AMBIENT_ROOTS = [98.00, 110.00, 123.47, 130.81]


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


def variant_url(url: str) -> str:
    """Ask Pollinations for a different seed of the same prompt -> a related
    but visually distinct second image, still 100% free."""
    sep = "&" if "?" in url else "?"
    return f"{url}{sep}seed={random.randint(1, 999999)}"


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


def generate_ambient_bed(duration: float, out_path: str, scene_index: int):
    """Synthesize a soft ambient pad (root + fifth + octave, slow tremolo,
    filtered) entirely with ffmpeg's built-in lavfi sources — no external
    audio files, no licensing concerns, no broken links."""
    root = AMBIENT_ROOTS[scene_index % len(AMBIENT_ROOTS)]
    fifth = root * 1.5
    octave = root * 2

    run([
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", f"sine=frequency={root:.2f}:duration={duration}:sample_rate=44100",
        "-f", "lavfi", "-i", f"sine=frequency={fifth:.2f}:duration={duration}:sample_rate=44100",
        "-f", "lavfi", "-i", f"sine=frequency={octave:.2f}:duration={duration}:sample_rate=44100",
        "-filter_complex",
        "[0:a]volume=0.15,tremolo=f=0.10:d=0.4[a0];"
        "[1:a]volume=0.10,tremolo=f=0.12:d=0.3[a1];"
        "[2:a]volume=0.08,tremolo=f=0.14:d=0.35[a2];"
        "[a0][a1][a2]amix=inputs=3:duration=longest:normalize=0,"
        "lowpass=f=1800,highpass=f=60[amb]",
        "-map", "[amb]",
        "-t", str(duration),
        "-ac", "2", "-ar", "44100",
        out_path,
    ])


def mix_audio(voice_path: str, ambient_path: str, out_path: str):
    """Mix the voiceover (full volume) with the ambient bed (quiet) so the
    narration stays clear while the background adds atmosphere."""
    run([
        "ffmpeg", "-y",
        "-i", voice_path, "-i", ambient_path,
        "-filter_complex",
        "[0:a]volume=1.0[v];[1:a]volume=0.18[a];"
        "[v][a]amix=inputs=2:duration=first:dropout_transition=2:normalize=0[mixed]",
        "-map", "[mixed]",
        "-c:a", "aac", "-b:a", "128k",
        out_path,
    ])


def movement_filter(pattern: int, total_frames: int, scale_width: int, scale_height: int) -> str:
    """Return a scale+zoompan filter string for one of 6 free, built-in
    ffmpeg camera movements, selected by pattern index."""
    pattern = pattern % 6
    last = max(total_frames - 1, 1)

    if pattern == 0:
        z, x, y = "zoom+0.0012", "iw/2-(iw/zoom/2)", "ih/2-(ih/zoom/2)"
    elif pattern == 1:
        z = "if(eq(on,0),1.15,max(zoom-0.0012,1.0))"
        x, y = "iw/2-(iw/zoom/2)", "ih/2-(ih/zoom/2)"
    elif pattern == 2:
        z = "1.1"
        x = f"(iw-iw/zoom)*(on/{last})"
        y = "ih/2-(ih/zoom/2)"
    elif pattern == 3:
        z = "1.1"
        x = f"(iw-iw/zoom)*(1-on/{last})"
        y = "ih/2-(ih/zoom/2)"
    elif pattern == 4:
        z = "zoom+0.0010"
        x = f"(iw-iw/zoom)*(on/{last})"
        y = f"(ih-ih/zoom)*(on/{last})"
    else:
        z = "1.05+0.0004*on"
        x, y = "iw/2-(iw/zoom/2)", "ih/2-(ih/zoom/2)"

    return (
        f"scale={scale_width}:{scale_height}:flags=fast_bilinear,"
        f"zoompan=z='{z}':x='{x}':y='{y}':d={total_frames}:s={WIDTH}x{HEIGHT}:fps={FPS}"
    )


def build_video_only_clip(image_path: str, duration: float, out_path: str, pattern: int):
    """Render one image as a short moving (video-only, no audio) clip."""
    total_frames = int(duration * FPS)
    scale_height = int(HEIGHT * 1.08)
    scale_width = int(WIDTH * 1.08)
    vf = movement_filter(pattern, total_frames, scale_width, scale_height)
    run([
        "ffmpeg", "-y",
        "-loop", "1", "-i", image_path,
        "-vf", vf,
        "-t", str(duration),
        "-pix_fmt", "yuv420p",
        "-c:v", "libx264", "-preset", "ultrafast", "-threads", "2",
        "-an",
        out_path,
    ])


def morph_two_clips(clip_a: str, clip_b: str, duration: float, out_path: str):
    """Cross-dissolve two same-length video-only clips across (almost) the
    full duration, producing a 'living' blend between two image variants."""
    fade_dur = max(duration - 0.15, 0.3)
    cmd = [
        "ffmpeg", "-y", "-i", clip_a, "-i", clip_b,
        "-filter_complex",
        f"[0:v][1:v]xfade=transition=fade:duration={fade_dur}:offset=0[v]",
        "-map", "[v]",
        "-c:v", "libx264", "-preset", "ultrafast", "-threads", "2",
        "-pix_fmt", "yuv420p",
        out_path,
    ]
    run(cmd)


def finalize_scene_clip(morphed_video: str, mixed_audio_path: str, srt_path: str, out_path: str):
    """Burn in captions + a light film-grain pass, then mux the final
    (voiceover + ambient) audio."""
    subtitle_style = (
        "FontName=Arial,FontSize=16,PrimaryColour=&H00FFFFFF,"
        "OutlineColour=&H00000000,BorderStyle=1,Outline=3,Shadow=0,"
        "Alignment=2,MarginV=120"
    )
    vf = f"subtitles={srt_path}:force_style='{subtitle_style}',noise=alls=6:allf=t+u"
    run([
        "ffmpeg", "-y",
        "-i", morphed_video, "-i", mixed_audio_path,
        "-vf", vf,
        "-c:v", "libx264", "-preset", "ultrafast", "-threads", "2",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "128k",
        "-shortest",
        out_path,
    ])


def concat_clips_with_crossfade(clip_paths: list, durations: list, out_path: str, workdir: str, fade_duration: float = 0.4):
    """Concatenate scene clips with a short crossfade between each."""
    if len(clip_paths) == 1:
        shutil.copy(clip_paths[0], out_path)
        return

    inputs = []
    for p in clip_paths:
        inputs += ["-i", p]

    filter_parts = []
    prev_label = "0:v"
    prev_audio = "0:a"
    running_offset = durations[0]

    for i in range(1, len(clip_paths)):
        v_out = f"v{i}"
        a_out = f"a{i}"
        offset = max(running_offset - fade_duration, 0)
        filter_parts.append(
            f"[{prev_label}][{i}:v]xfade=transition=fade:duration={fade_duration}:offset={offset}[{v_out}]"
        )
        filter_parts.append(
            f"[{prev_audio}][{i}:a]acrossfade=d={fade_duration}[{a_out}]"
        )
        prev_label = v_out
        prev_audio = a_out
        running_offset += durations[i] - fade_duration

    filter_complex = ";".join(filter_parts)

    cmd = ["ffmpeg", "-y"] + inputs + [
        "-filter_complex", filter_complex,
        "-map", f"[{prev_label}]", "-map", f"[{prev_audio}]",
        "-c:v", "libx264", "-preset", "ultrafast", "-threads", "2",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "96k",
        out_path,
    ]
    run(cmd)


def concat_clips_simple(clip_paths: list, out_path: str, workdir: str):
    """Fast, low-memory concat (hard cuts, no transition) — safety fallback."""
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
        durations = []

        for i, scene in enumerate(req.scenes):
            img_a_path = os.path.join(workdir, f"img_{i}a.jpg")
            img_b_path = os.path.join(workdir, f"img_{i}b.jpg")
            voice_path = os.path.join(workdir, f"voice_{i}.mp3")
            ambient_path = os.path.join(workdir, f"ambient_{i}.wav")
            mixed_audio_path = os.path.join(workdir, f"mixed_{i}.m4a")
            srt_path = os.path.join(workdir, f"cap_{i}.srt")
            clip_a_path = os.path.join(workdir, f"clip_{i}a.mp4")
            clip_b_path = os.path.join(workdir, f"clip_{i}b.mp4")
            morphed_path = os.path.join(workdir, f"morph_{i}.mp4")
            final_scene_path = os.path.join(workdir, f"scene_{i}.mp4")

            # 1. Two free image variants of the same prompt
            download_image(scene.image_url, img_a_path)
            download_image(variant_url(scene.image_url), img_b_path)

            # 2. Voiceover + ambient background + caption timing
            await generate_audio(scene.voiceover, req.voice, voice_path)
            duration = get_audio_duration(voice_path)
            generate_ambient_bed(duration, ambient_path, scene_index=i)
            mix_audio(voice_path, ambient_path, mixed_audio_path)
            build_srt(scene.voiceover, duration, srt_path)

            # 3. Two moving clips with different camera patterns
            build_video_only_clip(img_a_path, duration, clip_a_path, pattern=i)
            build_video_only_clip(img_b_path, duration, clip_b_path, pattern=i + 3)

            # 4. Morph between them, then burn captions + grain + mixed audio
            morph_two_clips(clip_a_path, clip_b_path, duration, morphed_path)
            finalize_scene_clip(morphed_path, mixed_audio_path, srt_path, final_scene_path)

            clip_paths.append(final_scene_path)
            durations.append(duration)

            # Free disk space from intermediate files as we go
            for p in (img_a_path, img_b_path, voice_path, ambient_path,
                      mixed_audio_path, clip_a_path, clip_b_path, morphed_path):
                try:
                    os.remove(p)
                except OSError:
                    pass

        final_path = os.path.join(workdir, "final.mp4")

        try:
            concat_clips_with_crossfade(clip_paths, durations, final_path, workdir)
        except Exception:
            concat_clips_simple(clip_paths, final_path, workdir)

        return FileResponse(
            final_path,
            media_type="video/mp4",
            filename="video.mp4",
            background=None,
        )

    except Exception as e:
        shutil.rmtree(workdir, ignore_errors=True)
        raise HTTPException(status_code=500, detail=f"Video generation failed: {str(e)}")
