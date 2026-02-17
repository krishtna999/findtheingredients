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
        # Man Sarvam's Batch API is the only thing that supports upto 1 hr data but there is no way to access the transcription via text directly lol ...
        # You gotta store the file and read it again :/
        job.download_outputs(output_dir=SARVAM_OUTPUT_DIR)

        # SDK's download_outputs saves as {input_file_name}.json
        input_file_name = file_results["successful"][0]["file_name"]
        transcription_file = os.path.join(SARVAM_OUTPUT_DIR, f"{input_file_name}.json")
        with open(transcription_file, "r") as f:
            transcription = f.read()
        return transcription
    else:
        raise Exception("Failed to transcribe audio")
