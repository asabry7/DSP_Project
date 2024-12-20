import streamlit as st
from allosaurus.app import read_recognizer
from pydub import AudioSegment
import io
import tempfile
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import scipy.io.wavfile as wavfile
from typing import List, Dict
import os

class PhonemeAnalyzer:
    def __init__(self, lang_id: str = "arb"):
        self.lang_id = lang_id
        self.recognizer = read_recognizer()

    def recognize_phonemes(self, file_path: str) -> List[Dict]:
        try:
            result = self.recognizer.recognize(file_path, lang_id=self.lang_id, timestamp=True)
            phoneme_data = []
            for entry in result.splitlines():
                parts = entry.split(" ")
                if len(parts) == 3:
                    phoneme_data.append({
                        "phoneme": parts[2],
                        "start": float(parts[0]),
                        "end": float(parts[1])
                    })
            return phoneme_data
        except Exception as e:
            st.error(f"Error recognizing phonemes for {file_path}: {e}")
            return []

    def align_sequences(self, teacher_phonemes: List[Dict], student_phonemes: List[Dict]):
        def dtw_distance(seq1, seq2):
            n, m = len(seq1), len(seq2)
            dtw_matrix = np.full((n + 1, m + 1), np.inf)
            dtw_matrix[0, 0] = 0

            for i in range(1, n + 1):
                for j in range(1, m + 1):
                    cost = 0 if seq1[i-1]['phoneme'] == seq2[j-1]['phoneme'] else 1
                    dtw_matrix[i, j] = cost + min(dtw_matrix[i-1, j], dtw_matrix[i, j-1], dtw_matrix[i-1, j-1])

            return dtw_matrix

        matrix = dtw_distance(teacher_phonemes, student_phonemes)
        aligned_teacher, aligned_student = [], []

        i, j = len(teacher_phonemes), len(student_phonemes)
        while i > 0 or j > 0:
            if i > 0 and j > 0 and matrix[i, j] == matrix[i-1, j-1]:
                aligned_teacher.insert(0, teacher_phonemes[i-1])
                aligned_student.insert(0, student_phonemes[j-1])
                i -= 1
                j -= 1
            elif i > 0 and (j == 0 or matrix[i, j] == matrix[i-1, j]):
                aligned_teacher.insert(0, teacher_phonemes[i-1])
                aligned_student.insert(0, {"phoneme": "-", "start": 0, "end": 0})
                i -= 1
            else:
                aligned_teacher.insert(0, {"phoneme": "-", "start": 0, "end": 0})
                aligned_student.insert(0, student_phonemes[j-1])
                j -= 1

        return aligned_teacher, aligned_student

    def analyze_differences(self, teacher_seq, student_seq):
        analysis = {
            'total_phonemes': len(teacher_seq),
            'correct_phonemes': 0,
            'errors': [],
            'missing_phonemes': [],
            'extra_phonemes': []
        }

        for t, s in zip(teacher_seq, student_seq):
            if t['phoneme'] == s['phoneme']:
                analysis['correct_phonemes'] += 1
            elif t['phoneme'] == '-':
                analysis['extra_phonemes'].append(s['phoneme'])
            elif s['phoneme'] == '-':
                analysis['missing_phonemes'].append(t['phoneme'])
            else:
                analysis['errors'].append((t['phoneme'], s['phoneme']))

        analysis['accuracy'] = analysis['correct_phonemes'] / analysis['total_phonemes'] * 100
        return analysis


# Initialize recognizer
@st.cache_data
def load_recognizer():
    model_directory = Path(".").resolve()
    universal_model_name = "uni2005"
    english_model_name = "eng2102"

    universal_recognizer = read_recognizer(
        inference_config_or_name=universal_model_name, alt_model_path=model_directory
    )
    english_recognizer = read_recognizer(
        inference_config_or_name=english_model_name, alt_model_path=model_directory
    )

    return universal_recognizer, english_recognizer


def preprocess_audio(utterance):
    audio_data = utterance.getbuffer()
    audio = AudioSegment.from_wav(io.BytesIO(audio_data))

    st.write(f"Duration: {audio.duration_seconds} seconds")
    st.write(f"Channels: {audio.channels}")
    st.write(f"Sample width: {audio.sample_width} bytes")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_wav_file:
        audio.export(temp_wav_file, format="wav")
        return temp_wav_file.name


def plot_spectrogram(audio_path):
    sample_rate, audio_data = wavfile.read(audio_path)
    if audio_data.ndim > 1:
        audio_data = audio_data[:, 0]  # Use only the first channel if stereo

    plt.figure(figsize=(10, 6))
    plt.specgram(audio_data, Fs=sample_rate, NFFT=2048, noverlap=1024, cmap='viridis')
    plt.title("Spectrogram")
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    st.pyplot(plt)
    plt.close()


def compare_phonemes(language, ref_audio_path, test_audio_path):
    analyzer = PhonemeAnalyzer(lang_id=language)
    st.write("### Reference Utterance's Spectogram")
    plot_spectrogram(ref_audio_path)
    st.write("### Test Utterance's Spectogram")
    plot_spectrogram(test_audio_path)

    ref_phonemes = analyzer.recognize_phonemes(ref_audio_path)
    test_phonemes = analyzer.recognize_phonemes(test_audio_path)

    if not ref_phonemes or not test_phonemes:
        st.error("Failed to recognize phonemes from one or both audio files.")
        return

    aligned_ref, aligned_test = analyzer.align_sequences(ref_phonemes, test_phonemes)
    analysis = analyzer.analyze_differences(aligned_ref, aligned_test)

    st.write("### Phoneme Comparison Analysis")
    st.write(f"Total Phonemes: {analysis['total_phonemes']}")
    st.write(f"Correct Phonemes: {analysis['correct_phonemes']}")
    st.write(f"Accuracy: {analysis['accuracy']:.2f}%")

    if analysis['errors']:
        st.write("### Errors")
        for expected, pronounced in analysis['errors']:
            st.write(f"Expected: {expected}, Pronounced: {pronounced}")

    if analysis['missing_phonemes']:
        st.write("### Missing Phonemes")
        st.write(", ".join(analysis['missing_phonemes']))

    if analysis['extra_phonemes']:
        st.write("### Extra Phonemes")
        st.write(", ".join(analysis['extra_phonemes']))

def plot_spectrogram(audio_path):
    sample_rate, audio_data = wavfile.read(audio_path)
    if audio_data.ndim > 1:
        audio_data = audio_data[:, 0]  # Use only the first channel if stereo

    plt.figure(figsize=(10, 6))
    plt.specgram(audio_data, Fs=sample_rate, NFFT=2048, noverlap=1024, cmap='viridis')
    plt.title("Spectrogram")
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    st.pyplot(plt)
    plt.close()


# Streamlit UI
st.title("DSP Project")

language = st.selectbox("Choose Language", ["eng", "arz", "arb"], format_func=lambda x: {
    "eng": "English",
    "arz": "Egyptian Arabic",
    "arb": "Arabic"
}[x])

st.write("### Record Reference Utterance")
reference_utterance = st.audio_input("Upload or record a reference utterance")

if reference_utterance is not None:
    reference_audio_path = preprocess_audio(reference_utterance)

    st.write("### Record Test Utterance")
    test_utterance = st.audio_input("Upload or record a test utterance")

    if test_utterance is not None:
        test_audio_path = preprocess_audio(test_utterance)
        compare_phonemes(language, reference_audio_path, test_audio_path)
