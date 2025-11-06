from google import genai
from google.genai import types
from gtts import gTTS
import wave
import streamlit as st
import pandas as pd
from PyPDF2 import PdfReader

# ---------- Helper: Save audio to .wav ----------
def wave_file(filename, pcm, channels=1, rate=24000, sample_width=2):
    with wave.open(filename, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(rate)
        wf.writeframes(pcm)

# ---------- Helper: Extract text from uploaded file ----------
def extract_text_from_file(uploaded_file):
    if uploaded_file is None:
        return None

    file_type = uploaded_file.type

    if "text" in file_type:
        return uploaded_file.read().decode("utf-8")

    elif "pdf" in file_type:
        reader = PdfReader(uploaded_file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        return text

    elif "csv" in file_type:
        df = pd.read_csv(uploaded_file)
        return df.to_string()

    elif "excel" in file_type or uploaded_file.name.endswith((".xlsx", ".xls")):
        df = pd.read_excel(uploaded_file)
        return df.to_string()

    else:
        return None

# ---------- Supported languages ----------
LANG_OPTIONS = {
    "English (US)": "en",
    "French": "fr",
    "Spanish": "es",
    "German": "de",
    "Italian": "it",
    "Chinese (Simplified)": "zh-CN",
    "Japanese": "ja",
    "Korean": "ko",
    "Igbo": "ig",
    "Yoruba": "yo",
    "Hausa": "ha",
    "Arabic": "ar",
    "Hindi": "hi",
    "Portuguese": "pt",
    "Russian": "ru"
}

# ---------- Streamlit Page Setup ----------
st.set_page_config(page_title="Text to Speech App", layout="centered")

st.markdown(
    "<h1 style='text-align: center; color: #4A90E2;'>Text to Speech with Gemini + gTTS Fallback</h1>",
    unsafe_allow_html=True
)

col1, col2, col3 = st.columns([1, 2, 1])
file_name = "out.wav"

# ---------- Main UI ----------
with col2:
    text_input = st.text_area("📝 Enter text:", placeholder="Type something here...", height=150)
    uploaded_file = st.file_uploader("📂 Or upload a file", type=["txt", "pdf", "xlsx", "csv"], accept_multiple_files=False)
    output_language = st.selectbox("🌍 Select output language:", list(LANG_OPTIONS.keys()))
    output_lang_code = LANG_OPTIONS[output_language]
    st.write("")

    # ---- Restrict to one input method ----
    if text_input and uploaded_file:
        st.warning("⚠️ Please either type text OR upload a file — not both.")
        st.stop()
    elif not text_input and not uploaded_file:
        st.info("ℹ️ Please provide text input or upload a file to continue.")
        st.stop()

    # Determine source of text
    text = extract_text_from_file(uploaded_file) if uploaded_file else text_input

    if st.button("🎧 Generate Audio"):
        with st.spinner("🔊 Generating audio... please wait"):
            try:
                client = genai.Client(api_key="AIzaSyAmSfmnnI0PfS69tFuIOsBC4REKzw5C2Kc")

                # Gemini TTS request
                response = client.models.generate_content(
                    model="gemini-2.0-flash",  # safer, general model
                    contents=f"Translate this text to {output_language} and provide audio output: {text}",
                    config=types.GenerateContentConfig(
                        response_modalities=["AUDIO"],
                        speech_config=types.SpeechConfig(
                            voice_config=types.VoiceConfig(
                                prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name="Callirrhoe")
                            )
                        ),
                    ),
                )

                # Check if valid audio response
                if not response or not getattr(response.candidates[0].content, "parts", None):
                    raise ValueError("Gemini returned no audio data.")

                data = response.candidates[0].content.parts[0].inline_data.data
                wave_file(file_name, data)
                st.success(f"✅ Audio generated successfully in {output_language} (Gemini)!")
                st.audio(file_name)

            except Exception as e:
                # ---- Fallback to gTTS ----
                st.warning(f"⚠️ Gemini error: {e}. Falling back to gTTS...")
                try:
                    tts = gTTS(text, lang=output_lang_code)
                    tts.save(file_name)
                    st.success(f"✅ Audio generated using gTTS in {output_language}!")
                    st.audio(file_name)
                except Exception as fallback_error:
                    st.error(f"❌ Fallback error: {fallback_error}")

# ---------- Footer ----------
st.markdown(
    "<p style='text-align: center; color: gray; font-size: 13px;'>Powered by Gemini & gTTS — with automatic fallback</p>",
    unsafe_allow_html=True
)




