#####################################       Start Start Start       ################################

import streamlit as st
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from allosaurus.app import read_recognizer
from pydub import AudioSegment
import io
import tempfile
from pathlib import Path
from scipy.spatial.distance import cdist

# Initialize recognizer (cached)
@st.cache_data
def LoadRecognizer():
    model_directory = Path(r".").resolve()  # Adjust path if needed
    universal_model_name = "uni2005"
    english_model_name = "eng2102"
    try:
        universal_recognizer = read_recognizer(inference_config_or_name=universal_model_name, alt_model_path=model_directory)
        english_recognizer = read_recognizer(inference_config_or_name=english_model_name, alt_model_path=model_directory)
        return universal_recognizer, english_recognizer
    except Exception as e:
        st.error(f"Error loading recognizers: {e}. Ensure Allosaurus models are available.")
        return None, None # Return None values if loading fails

def AudioPreprocessor(Utterance):
    audio_data = Utterance.getbuffer()
    audio = AudioSegment.from_wav(io.BytesIO(audio_data))
    st.write(f"Duration: {audio.duration_seconds} seconds")
    st.write(f"Channels: {audio.channels}")
    st.write(f"Sample width: {audio.sample_width} bytes")
    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_wav_file:
        temp_wav_file_path = temp_wav_file.name
        audio.export(temp_wav_file, format="wav")
        return temp_wav_file_path

def plot_spectrogram(audio_path, title):
    y, sr = librosa.load(audio_path)
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    plt.tight_layout()
    st.pyplot(plt)

def extract_mfcc(audio_path, sr=16000, n_mfcc=13, win_len=400, hop_len=160):
    y, sr = librosa.load(audio_path, sr=sr)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=win_len, hop_length=hop_len, window='hamming')
    return mfcc.T

def calculate_distance(mfcc1, mfcc2):
    """Calculates the cosine distance between two MFCC sequences."""
    min_len = min(len(mfcc1), len(mfcc2))
    distance = cdist(mfcc1[:min_len], mfcc2[:min_len], metric='cosine').diagonal() # Use diagonal for frame-wise comparison
    return distance

def plot_distance_with_phonemes(distance, threshold, ref_phonemes_with_time, test_phonemes_with_time):
    """Plots the distance with threshold and phoneme alignment."""

    num_frames = len(distance)
    frame_idx = np.arange(num_frames)
    time_per_frame = 0.01  # Assuming 10ms per frame (hop length)

    plt.figure(figsize=(12, 6))  # Increased figure height for phoneme labels

    # Plot the distance
    # Highlight sections of the curve above the threshold
    above_threshold = distance > threshold
    plt.plot(frame_idx, distance, label='Distance', color='blue')
    plt.fill_between(frame_idx, distance, threshold, where=above_threshold, color='red', alpha=0.5, label='Above Threshold')
    plt.axhline(y=threshold, color='r', linestyle='--', label=f'Threshold ({threshold:.2f})')

    # Add phoneme labels with timestamps
    y_offset = -0.1  # Adjust vertical offset for phoneme labels
    for phoneme_data in ref_phonemes_with_time:
        try:
            start_time, duration, phoneme = phoneme_data.split()
            start_time = float(start_time)
            end_time = start_time + float(duration)
            start_frame = int(start_time / time_per_frame)
            end_frame = int(end_time / time_per_frame)

            if 0 <= start_frame < num_frames:
                mid_frame = (start_frame + end_frame) // 2
                plt.text(mid_frame, y_offset, phoneme, ha='center', fontsize=16, color='blue')
        except ValueError:
            print(f"Skipping invalid phoneme data: {phoneme_data}")
            continue

    y_offset = -0.2  # Adjust offset for test phonemes
    for phoneme_data in test_phonemes_with_time:
        try:
            start_time, duration, phoneme = phoneme_data.split()
            start_time = float(start_time)
            end_time = start_time + float(duration)
            start_frame = int(start_time / time_per_frame)
            end_frame = int(end_time / time_per_frame)

            if 0 <= start_frame < num_frames:
                mid_frame = (start_frame + end_frame) // 2
                plt.text(mid_frame, y_offset, phoneme, ha='center', fontsize=16, color='green')
        except ValueError:
            print(f"Skipping invalid phoneme data: {phoneme_data}")
            continue

    plt.xlabel('Frame Index')
    plt.ylabel('Cosine Distance')
    plt.title('Frame-wise Cosine Distance with Phoneme Alignment')
    plt.legend()
    plt.tight_layout()
    st.pyplot(plt)


# Streamlit interface
st.title("DSP Project with MFCC and Phoneme Analysis")
LANGUAGE_ID = "eng"
language = st.selectbox("Choose Language", ["English", "Egyption Arabic", "Arabic"])

threshold = st.number_input("Enter Distance Threshold (0.0-1.0)", min_value=0.0, max_value=1.0, value=0.5, step=0.01)

# ReferenceUtterance = st.file_uploader("Upload a Reference Utterance", 'wav')
ReferenceUtterance = st.audio_input("Record a Reference Utterance")
if ReferenceUtterance is not None:
    reference_audio_path = AudioPreprocessor(ReferenceUtterance)
    plot_spectrogram(reference_audio_path, "Reference Utterance Spectrogram")

    # TestUtterance = st.file_uploader("Upload a Test Utterance", 'wav')
    TestUtterance = st.audio_input("Record a Test Utterance")
    if TestUtterance is not None:
        test_audio_path = AudioPreprocessor(TestUtterance)
        plot_spectrogram(test_audio_path, "Test Utterance Spectrogram")

        reference_mfcc = extract_mfcc(reference_audio_path)
        test_mfcc = extract_mfcc(test_audio_path)

        universal_recognizer, english_recognizer = LoadRecognizer()
        if universal_recognizer is None or english_recognizer is None:
            st.stop() # Stop execution if recognizers failed to load

        try:
            if language == "English":
                LANGUAGE_ID = "eng"
                ref_result = english_recognizer.recognize( reference_audio_path, LANGUAGE_ID)
                test_result = english_recognizer.recognize( test_audio_path, LANGUAGE_ID)
            else:
                if language == "Arabic":
                    LANGUAGE_ID = "arb"
                elif language == "Egyptian Arabic":
                    LANGUAGE_ID = "arz"
                elif language == "IPA":
                    LANGUAGE_ID = "ipa"
                ref_result = universal_recognizer.recognize( reference_audio_path, LANGUAGE_ID, timestamp=True)
                test_result = universal_recognizer.recognize( test_audio_path, LANGUAGE_ID, timestamp=True)


            st.write("### Reference phonemes:")
            st.write(ref_result)
            st.write("### Test phonemes:")
            st.write(test_result)

            # distance = calculate_distance(reference_mfcc, test_mfcc)
            # plot_distance_with_threshold(distance, threshold)
            distance = calculate_distance(reference_mfcc, test_mfcc)
            plot_distance_with_phonemes(distance, threshold, ref_result.strip().split('\n'), test_result.strip().split('\n')) # Pass the phonemes with timestamps

        except Exception as e:
            st.error(f"Error during recognition or alignment: {e}")