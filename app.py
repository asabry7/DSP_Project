import streamlit as st
from allosaurus.app import read_recognizer
from pydub import AudioSegment
import io
import tempfile
from pathlib import Path


# Initialize recognizer
@st.cache_data
def LoadRecognizer():
    # Path to the directory containing your model
    model_directory = Path(r".//").resolve()

    # Explicitly specify the model name if needed
    model_name = "uni2005"  # Replace with the correct model name
    recognizer = read_recognizer(inference_config_or_name=model_name, alt_model_path=model_directory)
    return recognizer


def AudioPreprocessor(Utterance):
    # Get the audio data from the BytesIO object
    audio_data = Utterance.getbuffer()

    # Read the audio data with pydub from the BytesIO object
    audio = AudioSegment.from_wav(io.BytesIO(audio_data))

    # Display basic audio info
    st.write(f"Duration: {audio.duration_seconds} seconds")
    st.write(f"Channels: {audio.channels}")
    st.write(f"Sample width: {audio.sample_width} bytes")

    # Create a temporary file to save the audio as a .wav file in memory
    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_wav_file:
        temp_wav_file_path = temp_wav_file.name
        # Export audio data to the temporary .wav file
        audio.export(temp_wav_file, format="wav")
        return temp_wav_file_path
    
# Title
st.title("DSP Project")

LANGUAGE_ID = "eng"
language = st.selectbox("Choose Language", ["English", "Egyption Arabic", "Arabic"])


# Record audio using audio_input
ReferenceUtterance = st.audio_input("Record a Reference Utterance")
# ReferenceUtterance = st.file_uploader("Record a Reference Utterance", type="wav")

if ReferenceUtterance is not None:
    # Get the audio data from the BytesIO object
    reference_audio_path = AudioPreprocessor(ReferenceUtterance)
    TestUtterance = st.audio_input("Record a Test Utterance")
    if TestUtterance is not None:
        if language == "Arabic":
            LANGUAGE_ID = "arb"
        elif language == "English":
            LANGUAGE_ID = "eng"
        elif language == "Egyption Arabic":
            LANGUAGE_ID = "arz"

        # Load the recognizer
        recognizer = LoadRecognizer()


        test_audio_path = AudioPreprocessor(TestUtterance)

        try:
            ref_result = recognizer.recognize(reference_audio_path, lang_id=LANGUAGE_ID)  # Use "arb" for Arabic
            test_result = recognizer.recognize(test_audio_path, lang_id=LANGUAGE_ID)  # Use "arb" for Arabic
            st.write("Reference Utterance Phonemes:")
            st.write(ref_result)
            st.write("Test Utterance Phonemes:")
            st.write(test_result)
        except Exception as e:
            st.error(f"Error during recognition: {e}")
    
   