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
import socket
import ipaddress
import asyncio
import subprocess
import requests
import edge_tts
from urllib.parse import urlparse
from fastapi import FastAPI, HTTPException, BackgroundTasks, Header, Depends
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Optional

app = FastAPI(title="Video Assembly API")

WIDTH, HEIGHT = 1080, 1920
FPS = 24
VOICE = "en-US-GuyNeural"
BGM_CACHE_PATH = "/tmp/bgm_pad_cache.mp3"
BGM_VOLUME = 0.12  # kept low so it never competes with the narration

# Set this in the environment (e.g. Railway variables) and send it back as
# the "X-API-Key" header from n8n. If it's left unset, auth is skipped -
# fine for local testing, but ALWAYS set this in production.
API_KEY = os.environ.get("VIDEO_API_KEY")

# Hosts/ranges an attacker could use to make this server fetch internal
# resources (cloud metadata endpoints, internal services, etc.) instead of
# a real image. Blocked before every image download.
_BLOCKED_HOSTS = {"localhost", "metadata.google.internal"}


def verify_api_key(x_api_key: Optional[str] = Header(default=None)):
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Missing or invalid X-API-Key header")


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


def _assert_public_url(url: str):
    """
    Raise if the URL's scheme isn't http(s), or its host resolves to a
    private/loopback/link-local address (blocks SSRF via internal services
    or cloud metadata endpoints like 169.254.169.254).
    """
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError(f"Unsupported URL scheme: {parsed.scheme!r}")
    host = parsed.hostname
    if not host or host.lower() in _BLOCKED_HOSTS:
        raise ValueError(f"Blocked host: {host!r}")
    try:
        addrs = {info[4][0] for info in socket.getaddrinfo(host, None)}
    except socket.gaierror as e:
        raise ValueError(f"Could not resolve host {host!r}: {e}")
    for addr in addrs:
        ip = ipaddress.ip_address(addr)
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast:
            raise ValueError(f"Blocked non-public IP for host {host!r}: {addr}")


def download_image(url: str, out_path: str):
    _assert_public_url(url)
    resp = requests.get(url, timeout=60, stream=True, allow_redirects=False)
    resp.raise_for_status()
    max_bytes = 25 * 1024 * 1024  # 25MB safety cap
    size = 0
    with open(out_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=65536):
            size += len(chunk)
            if size > max_bytes:
                raise ValueError("Downloaded image exceeds 25MB limit")
            f.write(chunk)


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
        "Style: Default,Arial,62,&H0000D7FF,&H00FFFFFF,&H60000000,&H00000000,-1,0,0,0,100,100,0,0,3,6,0,2,60,60,150,1\n\n"
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


def ensure_bgm_pad(path: str = BGM_CACHE_PATH) -> str:
    """
    Generate a short ambient chord-progression loop once, cached on disk,
    using only FFmpeg's built-in audio synthesis (sine oscillators, one
    triad per chord, tremolo + echo + lowpass for warmth). Fully synthetic -
    no external audio file, so zero licensing risk and zero cost. A simple
    Am-F-C-G progression (6s per chord, 24s total loop) sounds noticeably
    more musical than a single static drone, while still being cheap to
    generate and safe to loop via -stream_loop when mixed into the final
    video.
    """
    if os.path.exists(path):
        return path

    # Am, F, C, G triads (root, third, fifth), 6 seconds each.
    chords = [
        (110.00, 130.81, 164.81),  # A2, C3, E3  -> Am
        (87.31, 110.00, 130.81),   # F2, A2, C3  -> F
        (130.81, 164.81, 196.00),  # C3, E3, G3  -> C
        (98.00, 123.47,146.83),   # G2, B2, D3  -> G
    ]
    seg_dur = 6.0

    inputs = []
    for chord in chords:
        for freq in chord:
            inputs += ["-f", "lavfi", "-i", f"sine=frequency={freq}:duration={seg_dur}"]

    filter_parts = []
    chord_labels = []
    for i in range(len(chords)):
        i0, i1, i2 = 3 * i, 3 * i + 1, 3 * i + 2
        filter_parts.append(f"[{i0}:a]volume=0.24[c{i}n0]")
        filter_parts.append(f"[{i1}:a]volume=0.18[c{i}n1]")
        filter_parts.append(f"[{i2}:a]volume=0.14[c{i}n2]")
        filter_parts.append(f"[c{i}n0][c{i}n1][c{i}n2]amix=inputs=3:duration=longest[chord{i}]")
        chord_labels.append(f"[chord{i}]")

    filter_parts.append(f"{''.join(chord_labels)}concat=n={len(chords)}:v=0:a=1[progression]")
    filter_parts.append(
        "[progression]tremolo=f=0.15:d=0.3,"
        "aecho=0.6:0.5:120:0.3,"
        "lowpass=f=2600,volume=2.2[out]"
    )
    filter_complex = ";".join(filter_parts)

    run([
        "ffmpeg", "-y",
        *inputs,
        "-filter_complex", filter_complex,
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


@app.post("/create-video", dependencies=[Depends(verify_api_key)])
async def create_video(req: VideoRequest, background_tasks: BackgroundTasks):
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

        # Clean up the workdir only after the file has actually been
        # streamed back to the client (FileResponse needs it to still
        # exist on disk while sending).
        background_tasks.add_task(shutil.rmtree, workdir, ignore_errors=True)

        return FileResponse(
            final_path,
            media_type="video/mp4",
            filename="video.mp4",
        )

    except Exception as e:
        shutil.rmtree(workdir, ignore_errors=True)
        raise HTTPException(status_code=500, detail=f"Video generation failed: {str(e)}")

app = FastAPI(title="Video Assembly API")

WIDTH, HEIGHT = 1080, 1920
FPS = 20
VOICE = "en-US-GuyNeural"
CRF = "26"                       # slightly smaller/lighter encodes, minimal visible quality loss
DOWNLOAD_WIDTH, DOWNLOAD_HEIGHT = 810, 1440   # request smaller source images (still upscaled to 1080x1920 on render)
MAX_SCENES = 8                   # soft cap to keep total render time/memory bounded
AMBIENT_MASTER_DURATION = 40     # seconds — long enough to cover any single scene, generated once per root note

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


def force_dimensions(url: str, width: int = DOWNLOAD_WIDTH, height: int = DOWNLOAD_HEIGHT) -> str:
    """Override the width/height query params on a Pollinations URL so we
    always download a smaller source image, regardless of what the caller
    requested — cuts network + decode memory noticeably on tight RAM hosts."""
    parsed = urlparse(url)
    qs = parse_qs(parsed.query)
    qs["width"] = [str(width)]
    qs["height"] = [str(height)]
    new_query = urlencode(qs, doseq=True)
    return urlunparse(parsed._replace(query=new_query))


def download_image(url: str, out_path: str):
    resp = requests.get(force_dimensions(url), timeout=60)
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


def generate_ambient_master(root: float, out_path: str):
    """Synthesize one long ambient pad (root + fifth + octave, slow tremolo,
    filtered) using ffmpeg's built-in lavfi sources. Rendered ONCE per root
    note and then cheaply trimmed per scene (see trim_ambient), instead of
    re-synthesizing from scratch for every single scene."""
    fifth = root * 1.5
    octave = root * 2
    run([
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", f"sine=frequency={root:.2f}:duration={AMBIENT_MASTER_DURATION}:sample_rate=44100",
        "-f", "lavfi", "-i", f"sine=frequency={fifth:.2f}:duration={AMBIENT_MASTER_DURATION}:sample_rate=44100",
        "-f", "lavfi", "-i", f"sine=frequency={octave:.2f}:duration={AMBIENT_MASTER_DURATION}:sample_rate=44100",
        "-filter_complex",
        "[0:a]volume=0.15,tremolo=f=0.10:d=0.4[a0];"
        "[1:a]volume=0.10,tremolo=f=0.07:d=0.3[a1];"
        "[2:a]volume=0.08,tremolo=f=0.13:d=0.35[a2];"
        "[a0][a1][a2]amix=inputs=3:duration=longest:normalize=0,"
        "lowpass=f=1800,highpass=f=60[amb]",
        "-map", "[amb]",
        "-t", str(AMBIENT_MASTER_DURATION),
        "-ac", "2", "-ar", "44100",
        out_path,
    ])


def trim_ambient(master_path: str, duration: float, out_path: str):
    """Cheaply cut a scene-length chunk out of the pre-rendered ambient
    master — no re-synthesis, just a fast stream copy."""
    run([
        "ffmpeg", "-y", "-i", master_path,
        "-t", str(duration),
        "-c", "copy",
        out_path,
    ])


def mix_audio(voice_path: str, ambient_path: str, out_path: str):
    """Mix the voiceover (full volume) with the ambient bed (quiet) so the
    narration stays clear while the background adds atmosphere."""
    run([
        "ffmpeg", "-y",
        "-i", voice_path, "-i", ambient_path,
        "-filter_complex",
        "[0:a]volume=1.0[v];[1:a]volume=0.21[a];"
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
        "-c:v", "libx264", "-preset", "ultrafast", "-crf", CRF, "-threads", "2",
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
        "-c:v", "libx264", "-preset", "ultrafast", "-crf", CRF, "-threads", "2",
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
        "-c:v", "libx264", "-preset", "ultrafast", "-crf", CRF, "-threads", "2",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "128k",
        "-max_muxing_queue_size", "1024",
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
        "-c:v", "libx264", "-preset", "ultrafast", "-crf", CRF, "-threads", "2",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "96k",
        "-max_muxing_queue_size", "1024",
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

    scenes = req.scenes[:MAX_SCENES]  # soft cap — keeps total render time/memory bounded on 1GB RAM

    job_id = str(uuid.uuid4())
    workdir = f"/tmp/video_{job_id}"
    os.makedirs(workdir, exist_ok=True)

    ambient_masters = {}  # root_frequency -> master file path, built once and reused

    try:
        clip_paths = []
        durations = []

        for i, scene in enumerate(scenes):
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

            # 1. Two free image variants of the same prompt (downsized for lower memory use)
            download_image(scene.image_url, img_a_path)
            download_image(variant_url(scene.image_url), img_b_path)

            # 2. Voiceover + ambient background (cached master, cheaply trimmed) + captions
            await generate_audio(scene.voiceover, req.voice, voice_path)
            duration = get_audio_duration(voice_path)

            root = AMBIENT_ROOTS[i % len(AMBIENT_ROOTS)]
            if root not in ambient_masters:
                master_path = os.path.join(workdir, f"ambient_master_{i % len(AMBIENT_ROOTS)}.wav")
                generate_ambient_master(root, master_path)
                ambient_masters[root] = master_path
            trim_ambient(ambient_masters[root], duration, ambient_path)
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

            # Brief pause so the OS can reclaim memory between scenes —
            # cheap insurance against OOM kills on a 1GB RAM host.
            time.sleep(1)

        # Ambient masters no longer needed once every scene has trimmed from them
        for p in ambient_masters.values():
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
