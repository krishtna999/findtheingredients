import yt_dlp
import os

AUDIO_DIR = "saved_audios"


def fetch_metadata(video_url: str):
    ydl_opts = {"skip_download": True, "quiet": True, "extract_flat": False}
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(video_url, download=False)

    return {
        "title": info.get("title"),
        "tags": info.get("tags"),
        "language": info.get("language"),
        "description": info.get("description"),
    }


def download_audio(video_url: str, video_title: str):

    os.makedirs(AUDIO_DIR, exist_ok=True)
    output_path = f"{AUDIO_DIR}/{video_title}"
    audio_file = f"{output_path}.mp3"

    if os.path.exists(audio_file):
        print(f"[youtube] Using cached audio for '{video_title}'")
        return audio_file

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": output_path,
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "128",
            }
        ],
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([video_url])

    # Add extension. At the mercy of the yt_dlp library.
    output_path += ".mp3"

    return output_path
