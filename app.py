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

# Title
st.title("DSP Project")

# Record audio using audio_input
uploaded_audio = st.audio_input("Record a Reference Utterance")
# uploaded_audio = st.file_uploader("Record a Reference Utterance", type="wav")

if uploaded_audio is not None:
    # Load the recognizer
    recognizer = LoadRecognizer()

    # Get the audio data from the BytesIO object
    audio_data = uploaded_audio.getbuffer()

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
        
        # st.write(f"Audio saved temporarily as: {temp_wav_file_path}")

        # Now you can use the temporary file path with the recognizer
        try:
            result = recognizer.recognize(temp_wav_file_path, lang_id="arb")  # Use "arb" for Arabic
            st.write("Recognition Result:")
            st.write(result)
        except Exception as e:
            st.error(f"Error during recognition: {e}")
    
   