from langchain.tools import tool
from ingredient_finder.services.transcription.sarvam import translate_audio


@tool
def transcribe_audio_indian(audio_path: str):
    """
    Use only for Indian languages (supports code-mixing as well). Transcribes and translates the audio from `audio_path` to English.
    Returns the transcribed and translated text.
    """
    return translate_audio(audio_path)


# ADR: Using Sarvam AI again cuz I'm feeling a bit lazy to implement another transcription API...
# But it is a separate tool cuz I wanna see how the Agent reasons.
@tool
def transcribe_audio_english(audio_path: str):
    """
    Use only for pure English audio. Transcribes the audio from `audio_path` to English.
    Returns the transcribed text.
    """
    return translate_audio(audio_path)


transcription_tools = [
    transcribe_audio_indian,
    transcribe_audio_english,
]

transcription_tools_by_name = {tool.name: tool for tool in transcription_tools}
