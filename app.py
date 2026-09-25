import os
import uuid
import subprocess
from flask import Flask, request, jsonify
import replicate  # requires REPLICATE_API_TOKEN in env

app = Flask(__name__)

UPLOAD_DIR = "/tmp/face-mimic"
os.makedirs(UPLOAD_DIR, exist_ok=True)

ALLOWED_IMAGE_EXT = {".png", ".jpg", ".jpeg"}
ALLOWED_AUDIO_EXT = {".mp3", ".wav", ".m4a", ".aac"}
ALLOWED_VIDEO_EXT = {".mp4", ".mov"}

MAX_FILE_SIZE_MB = 25


def _ext(filename):
    return os.path.splitext(filename)[1].lower()


def _save_upload(file_storage, allowed_exts):
    ext = _ext(file_storage.filename)
    if ext not in allowed_exts:
        raise ValueError(f"Unsupported file type: {ext}")
    path = os.path.join(UPLOAD_DIR, f"{uuid.uuid4()}{ext}")
    file_storage.save(path)
    size_mb = os.path.getsize(path) / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        os.remove(path)
        raise ValueError("File too large")
    return path


def _extract_audio_from_video(video_path):
    """If the user recorded a video (photo talking along to their own video),
    pull the audio track out with ffmpeg so it can be fed to the reenactment model."""
    audio_path = video_path + ".wav"
    subprocess.run(
        ["ffmpeg", "-y", "-i", video_path, "-vn", "-acodec", "pcm_s16le", audio_path],
        check=True,
        capture_output=True,
    )
    return audio_path


def _add_watermark(video_url):
    """
    Stamp a visible 'AI-generated' watermark onto the output video.
    Placeholder: in production, download video_url, burn in text with ffmpeg
    (drawtext filter), re-upload/serve, and return the new URL.
    Left unimplemented here since it depends on where you serve output files from.
    """
    return video_url


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


@app.route("/animate-face", methods=["POST"])
def animate_face():
    # --- Consent gate ---
    # The client must explicitly confirm this is the user's own photo/likeness,
    # or one they have permission to animate. Refuse to process without it.
    consent = request.form.get("consent_own_photo")
    if consent != "true":
        return jsonify({
            "status": "error",
            "message": "Consent required: you must confirm you own or have permission to use this photo."
        }), 403

    if "photo" not in request.files:
        return jsonify({"status": "error", "message": "Missing photo file"}), 400

    has_audio = "audio" in request.files
    has_video = "video" in request.files
    if not has_audio and not has_video:
        return jsonify({"status": "error", "message": "Provide either an audio or a video recording"}), 400

    photo_path = audio_path = video_path = extracted_audio_path = None

    try:
        photo_path = _save_upload(request.files["photo"], ALLOWED_IMAGE_EXT)

        if has_video:
            video_path = _save_upload(request.files["video"], ALLOWED_VIDEO_EXT)
            audio_path = _extract_audio_from_video(video_path)
            extracted_audio_path = audio_path
        else:
            audio_path = _save_upload(request.files["audio"], ALLOWED_AUDIO_EXT)

        with open(photo_path, "rb") as pf, open(audio_path, "rb") as af:
            output = replicate.run(
                # LivePortrait or SadTalker both work here; SadTalker shown as the
                # known-working option from the earlier prototype.
                "cjwbw/sadtalker:a22213e414168019e09d84e55e8c253e7f411261d7a8d5f3089c89078f135b1d",
                input={
                    "source_image": pf,
                    "driven_audio": af,
                    "still": False,
                    "enhancer": "gfpgan",
                },
            )

        watermarked_url = _add_watermark(output)

        return jsonify({
            "status": "success",
            "video_url": watermarked_url,
            "note": "AI-generated content. Watermarking should be verified before public use.",
        })

    except ValueError as e:
        return jsonify({"status": "error", "message": str(e)}), 400
    except subprocess.CalledProcessError:
        return jsonify({"status": "error", "message": "Failed to process video/audio"}), 500
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500
    finally:
        for p in (photo_path, audio_path, video_path, extracted_audio_path):
            if p and os.path.exists(p):
                os.remove(p)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)