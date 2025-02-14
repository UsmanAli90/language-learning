import whisper
from gtts import gTTS
from fuzzywuzzy import fuzz
import streamlit as st
import os

model = whisper.load_model("tiny")

def generate_native_audio(phrase, language="en"):
    """Generate a native speaker audio for a given phrase."""
    tts = gTTS(text=phrase, lang=language, slow=False)
    output_file = "native_audio.mp3"
    tts.save(output_file)
    return output_file

def evaluate_pronunciation(user_audio_path, target_phrase):
    """
    Evaluate the user's pronunciation:
    - Transcribe the user's audio.
    - Compare it with the target phrase using fuzzy string matching.
    """
    transcription_result = model.transcribe(user_audio_path)
    user_transcription = transcription_result["text"]

    similarity_score = fuzz.ratio(user_transcription.lower(), target_phrase.lower())

    feedback = f"Your transcription: {user_transcription}\n"
    feedback += f"Similarity with target phrase: {similarity_score}%"
    return feedback

st.title("Voice-Driven Language Learning App")
st.markdown(
    "Practice speaking phrases in a foreign language and get instant feedback on your pronunciation."
)

phrase_to_practice = st.text_input("Enter the phrase to practice:", value="Hello, how are you?")
language = st.selectbox("Select language:", ["en", "es", "fr", "de"])

if st.button("Generate Native Pronunciation"):
    native_audio_path = generate_native_audio(phrase_to_practice, language)
    audio_file = open(native_audio_path, "rb")
    audio_bytes = audio_file.read()
    st.audio(audio_bytes, format="audio/mp3", start_time=0)

st.markdown("### Record and upload your pronunciation:")
uploaded_file = st.file_uploader("Upload your audio file:", type=["mp3", "wav", "m4a"])

if uploaded_file is not None:
    with open("uploaded_audio.mp3", "wb") as f:
        f.write(uploaded_file.read())
    target_audio_path = "uploaded_audio.mp3"

    if st.button("Evaluate Pronunciation"):
        feedback = evaluate_pronunciation(target_audio_path, phrase_to_practice)
        st.text_area("Feedback:", feedback)

