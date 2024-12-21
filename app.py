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
from fastdtw import fastdtw


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
    st.warning(f"Duration: {audio.duration_seconds} seconds")
    st.warning(f"Channels: {audio.channels}")
    st.warning(f"Sample width: {audio.sample_width} bytes")
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

def calculate_edit_distance_with_details(text1, text2):
    """Calculates edit distance and returns details of operations."""
    len1 = len(text1)
    len2 = len(text2)

    dp = np.zeros((len1 + 1, len2 + 1), dtype=int)
    ops = [["" for _ in range(len2 + 1)] for _ in range(len1 + 1)] # Store operations

    for i in range(len1 + 1):
        dp[i][0] = i
        ops[i][0] = "D" * i # Deletion
    for j in range(len2 + 1):
        dp[0][j] = j
        ops[0][j] = "I" * j # Insertion

    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            if text1[i - 1] == text2[j - 1]:
                cost = 0
                op = ""
            else:
                cost = 1
                op = "S" # Substitution

            dp[i][j] = min(
                dp[i - 1][j - 1] + cost,  # Substitution/Match
                dp[i - 1][j] + 1,      # Deletion
                dp[i][j - 1] + 1       # Insertion
            )

            if dp[i][j] == dp[i - 1][j - 1] + cost:
                ops[i][j] = ops[i-1][j-1] + op
            elif dp[i][j] == dp[i - 1][j] + 1:
                ops[i][j] = ops[i-1][j] + "D"
            else:
                ops[i][j] = ops[i][j-1] + "I"
                
    return dp[len1][len2], ops[len1][len2]


def calculate_distance(mfcc1, mfcc2):
    """Calculates the cosine distance between two MFCC sequences."""
    min_len = min(len(mfcc1), len(mfcc2))
    distance = cdist(mfcc1[:min_len], mfcc2[:min_len], metric='cosine').diagonal() # Use diagonal for frame-wise comparison
    return distance

def align_phonemes_with_dtw(ref_mfcc, test_mfcc, ref_phonemes_with_time, test_phonemes_with_time):
    """Aligns phonemes using fastdtw and returns aligned phoneme sequences."""

    distance, path = fastdtw(ref_mfcc, test_mfcc, dist=lambda x, y: np.linalg.norm(x - y))

    # Get aligned frame indices (path is a list of tuples)
    ref_aligned_frames = [p[0] for p in path]
    test_aligned_frames = [p[1] for p in path]

    ref_aligned_phonemes = []
    test_aligned_phonemes = []

    time_per_frame = 0.01  # Assuming 10ms per frame

    for ref_frame, test_frame in path:
      ref_time = ref_frame * time_per_frame
      test_time = test_frame * time_per_frame

      # Find corresponding phonemes based on time
      ref_phoneme = ""
      for phoneme_data in ref_phonemes_with_time:
        start_time, duration, phoneme = phoneme_data.split()
        start_time = float(start_time)
        end_time = start_time + float(duration)
        if start_time <= ref_time <= end_time:
          ref_phoneme = phoneme
          break
      
      test_phoneme = ""
      for phoneme_data in test_phonemes_with_time:
        start_time, duration, phoneme = phoneme_data.split()
        start_time = float(start_time)
        end_time = start_time + float(duration)
        if start_time <= test_time <= end_time:
          test_phoneme = phoneme
          break
      
      ref_aligned_phonemes.append(ref_phoneme)
      test_aligned_phonemes.append(test_phoneme)
    
    return ref_aligned_phonemes, test_aligned_phonemes, ref_aligned_frames, test_aligned_frames


def plot_distance_with_aligned_phonemes(distance, threshold, ref_aligned_phonemes, test_aligned_phonemes, ref_aligned_frames, test_aligned_frames):
    plt.figure(figsize=(12, 6))
    above_threshold = distance > threshold
    plt.plot(np.arange(len(distance)), distance, label='Distance', color='blue')
    plt.fill_between(np.arange(len(distance)), distance, threshold, where=above_threshold, color='red', alpha=0.5, label='Above Threshold')
    plt.axhline(y=threshold, color='r', linestyle='--', label=f'Threshold ({threshold:.2f})')
    
    y_offset = -0.1
    last_ref_phoneme = None
    for i, phoneme in enumerate(ref_aligned_phonemes):
        if phoneme != "" and phoneme != last_ref_phoneme:
            plt.text(i, y_offset, f"/{phoneme}/", ha='center', fontsize=16, color='blue')
            last_ref_phoneme = phoneme
        elif phoneme != "":  # Add space if the same phoneme repeats
            plt.text(i, y_offset, f" ", ha='center', fontsize=16, color='blue')
            
    
    y_offset = -0.2
    last_test_phoneme = None
    for i, phoneme in enumerate(test_aligned_phonemes):
        if phoneme != "" and phoneme != last_test_phoneme:
            plt.text(i, y_offset, f"/{phoneme}/", ha='center', fontsize=16, color='green')
            last_test_phoneme = phoneme
        elif phoneme != "":
            plt.text(i, y_offset, f" ", ha='center', fontsize=16, color='green')

    plt.xlabel('Frame Index')
    plt.ylabel('Cosine Distance')
    plt.title('Frame-wise Cosine Distance with Phoneme Alignment')
    plt.legend()
    plt.tight_layout()
    st.pyplot(plt)


hide_streamlit_style = """
            <style>
                /* Hide the Streamlit header and menu */
                header {visibility: hidden;}
                /* Optionally, hide the footer */
                .streamlit-footer {display: none;}
                /* Hide your specific div class, replace class name with the one you identified */
                .st-emotion-cache-uf99v8 {display: none;}
            </style>
            """

st.markdown(hide_streamlit_style, unsafe_allow_html=True)
col1, col2, col3 = st.columns(3)

with col1:
    st.write(' ')

with col2:
    st.image("Logo.jpg", width = 200)
    

with col3:
    st.write(' ')
    
# Streamlit interface
st.write("### Don't Escape from Excellence, Dive into AccentCave")
st.divider()


# st.title("AccentCave")
LANGUAGE_ID = "eng"
language = st.selectbox("Choose Language", ["English", "Egyption Arabic", "Arabic"])

threshold = st.number_input("Enter Distance Threshold (0.0-1.0)", min_value=0.0, max_value=1.0, value=0.5, step=0.01)

ref_method = st.radio("Input Reference Audio?", ["Record", "Upload a .wav file"])
if ref_method == "Record":
    ReferenceUtterance = st.audio_input("Record a Reference Utterance")
elif ref_method == "Upload a .wav file":
    ReferenceUtterance = st.file_uploader("Upload a Reference Utterance", 'wav')
# 
if ReferenceUtterance is not None:
    reference_audio_path = AudioPreprocessor(ReferenceUtterance)
    plot_spectrogram(reference_audio_path, "Reference Utterance Spectrogram")

    test_method = st.radio("Input Test Audio?", ["Record", "Upload a .wav file"])
    if test_method == "Record":
        TestUtterance = st.audio_input("Record a Test Utterance")
    elif test_method == "Upload a .wav file":
        TestUtterance = st.file_uploader("Upload a Test Utterance", 'wav')

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
                ref_result = english_recognizer.recognize( reference_audio_path, LANGUAGE_ID, timestamp=True)
                test_result = english_recognizer.recognize( test_audio_path, LANGUAGE_ID, timestamp=True)
            else:
                if language == "Arabic":
                    LANGUAGE_ID = "arb"
                elif language == "Egyptian Arabic":
                    LANGUAGE_ID = "arz"
                ref_result = universal_recognizer.recognize( reference_audio_path, LANGUAGE_ID, timestamp=True)
                test_result = universal_recognizer.recognize( test_audio_path, LANGUAGE_ID, timestamp=True)


            ref_text = " ".join([line.split()[2] for line in ref_result.strip().split("\n")])
            test_text = " ".join([line.split()[2] for line in test_result.strip().split("\n")])

            edit_distance, operations = calculate_edit_distance_with_details(ref_text, test_text)

            st.write("### Edit Distance and Operations:")
            st.write(f"Reference: {ref_text}")
            st.write(f"Test: {test_text}")
            st.write(f"Distance: {edit_distance}")

            st.write(f"Operations: {operations}")

            # Detailed Operations Breakdown
            insertions = operations.count("I")
            deletions = operations.count("D")
            substitutions = operations.count("S")

            st.write("### Operation Breakdown:")
            st.write(f"Insertions: {insertions}")
            st.write(f"Deletions: {deletions}")
            st.write(f"Substitutions: {substitutions}")


            # distance = calculate_distance(reference_mfcc, test_mfcc)
            # plot_distance_with_threshold(distance, threshold)
            distance = calculate_distance(reference_mfcc, test_mfcc)

            ref_aligned_phonemes, test_aligned_phonemes, ref_aligned_frames, test_aligned_frames = align_phonemes_with_dtw(reference_mfcc, test_mfcc, ref_result.strip().split('\n'), test_result.strip().split('\n'))
            plot_distance_with_aligned_phonemes(distance, threshold, ref_aligned_phonemes, test_aligned_phonemes, ref_aligned_frames, test_aligned_frames)
        except Exception as e:
            st.error(f"Error during recognition or alignment: {e}")