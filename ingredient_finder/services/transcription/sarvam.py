from sarvamai import SarvamAI
import os

SARVAM_OUTPUT_DIR = "transcription/sarvam_outputs"


def translate_audio(audio_path: str):
    os.makedirs(SARVAM_OUTPUT_DIR, exist_ok=True)
    client = SarvamAI(api_subscription_key=os.getenv("SARVAM_API_KEY"))

    # Create batch job — change mode as needed
    job = client.speech_to_text_job.create_job(
        model="saaras:v3",
        mode="translate",
        language_code="unknown",
        with_diarization=False,
    )

    # Upload and process files
    audio_paths = [audio_path]
    job.upload_files(file_paths=audio_paths)
    job.start()

    # Wait for completion
    job.wait_until_complete()

    # Check file-level results
    file_results = job.get_file_results()

    # Download outputs for successful files
    if file_results["successful"]:
        job.download_outputs(output_dir=SARVAM_OUTPUT_DIR)

        # Read the downloaded transcription file
        output_files = os.listdir(SARVAM_OUTPUT_DIR)
        if output_files:
            # Assuming the transcription is in a txt or json file
            transcription_file = os.path.join(SARVAM_OUTPUT_DIR, output_files[0])
            with open(transcription_file, "r") as f:
                transcription = f.read()
            return transcription
        else:
            raise Exception("No output files found in directory")
    else:
        raise Exception("Failed to transcribe audio")
