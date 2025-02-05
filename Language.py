import whisper
from gtts import gTTS
from fuzzywuzzy import fuzz
import gradio as gr
import os

model = whisper.load_model("tiny")

def generate_native_audio(phrase, language="en"):
    """Generate a native speaker audio for a given phrase."""
    tts = gTTS(text=phrase, lang=language, slow=False)
    output_file = "native_audio.mp3"
    tts.save(output_file)
    return output_file

def evaluate_pronunciation(user_audio, target_phrase):
    """
    Evaluate the user's pronunciation:
    - Transcribe the user's audio.
    - Compare it with the target phrase using fuzzy string matching.
    """
    transcription_result = model.transcribe(user_audio)
    user_transcription = transcription_result["text"]

    similarity_score = fuzz.ratio(user_transcription.lower(), target_phrase.lower())

    feedback = f"Your transcription: {user_transcription}\n"
    feedback += f"Similarity with target phrase: {similarity_score}%"
    return feedback

def language_learning_app(phrase, language="en"):
    """
    Main Gradio application for the Voice-Driven Language Learning App.
    """

    native_audio = generate_native_audio(phrase, language)

    with gr.Blocks() as app:
        gr.Markdown("# Voice-Driven Language Learning App 🌍🎙️")
        gr.Markdown(
            "Practice speaking phrases in a foreign language and get instant feedback on your pronunciation."
        )

        gr.Markdown("### Listen to the native pronunciation:")
        audio_player = gr.Audio(native_audio, autoplay=False, label="Native Pronunciation")

        gr.Markdown("### Record your pronunciation:")
        with gr.Row():
            user_audio_input = gr.Audio(type="filepath", label="Your Audio")
            evaluate_button = gr.Button("Evaluate Pronunciation")

        result_output = gr.Textbox(label="Feedback")

        evaluate_button.click(
            evaluate_pronunciation,
            inputs=[user_audio_input, gr.Textbox(value=phrase, label="Target Phrase")],
            outputs=result_output,
        )

    return app

phrase_to_practice = "Hello How are you"
language_learning_app(phrase_to_practice, language="en").launch()
