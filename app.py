import streamlit as st
from allosaurus.app import read_recognizer
from pydub import AudioSegment
import io
import tempfile
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional

# Initialize recognizer
@st.cache_data
def LoadRecognizer():
    model_directory = Path(r".//").resolve()
    universal_model_name = "uni2005"  
    english_model_name = "eng2102"  
    universal_recognizer = read_recognizer(inference_config_or_name=universal_model_name, alt_model_path=model_directory)
    english_recognizer = read_recognizer(inference_config_or_name=english_model_name, alt_model_path=model_directory)
    return universal_recognizer, english_recognizer

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

def plot_spectrogram(audio_path):
    from scipy.io import wavfile
    from scipy.signal import spectrogram

    sample_rate, audio_data = wavfile.read(audio_path)
    frequencies, times, spectrogram_data = spectrogram(audio_data, fs=sample_rate)

    plt.figure(figsize=(10, 6))
    plt.imshow(10 * np.log10(spectrogram_data), aspect='auto', cmap='viridis', origin='lower', 
               extent=[times.min(), times.max(), frequencies.min(), frequencies.max()])
    plt.colorbar(label='Power (dB)')
    plt.title('Spectrogram')
    plt.ylabel('Frequency (Hz)')
    plt.xlabel('Time (s)')
    st.pyplot(plt)
    plt.close()

def recognize_phonemes(recognizer, file_path: str, lang_id: str) -> Optional[List[Dict]]:
    if not Path(file_path).exists():
        st.error(f"Error: File {file_path} not found.")
        return None

    try:
        result = recognizer.recognize(file_path, lang_id=lang_id, timestamp=True)
        phoneme_data = []
        for entry in result.splitlines():
            parts = entry.split(" ")
            if len(parts) == 3:
                start_time = float(parts[0])
                end_time = float(parts[1])
                phoneme = parts[2]
                phoneme_data.append({"phoneme": phoneme, "start": start_time, "end": end_time})
        return phoneme_data
    except Exception as e:
        st.error(f"Error processing {file_path}: {str(e)}")
        return None

def align_sequences(teacher_phonemes: List[Dict], student_phonemes: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
    def dtw_distance(seq1, seq2):
        n, m = len(seq1), len(seq2)
        dtw_matrix = np.full((n + 1, m + 1), np.inf)
        dtw_matrix[0, 0] = 0

        for i in range(1, n + 1):
            for j in range(1, m + 1):
                cost = 0 if seq1[i-1].get('phoneme') == seq2[j-1].get('phoneme') else 1
                dtw_matrix[i, j] = cost + min(dtw_matrix[i-1, j],
                                              dtw_matrix[i, j-1],
                                              dtw_matrix[i-1, j-1])

        return dtw_matrix

    matrix = dtw_distance(teacher_phonemes, student_phonemes)
    aligned_teacher = []
    aligned_student = []

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

def analyze_differences(teacher_seq: List[Dict], student_seq: List[Dict]) -> Dict:
    analysis = {
        'total_phonemes': len(teacher_seq),
        'correct_phonemes': 0,
        'errors': [],
        'timing_differences': [],
        'missing_phonemes': [],
        'extra_phonemes': [],
        'accuracy_percentage': 0
    }

    for i, (teacher, student) in enumerate(zip(teacher_seq, student_seq)):
        if teacher['phoneme'] == student['phoneme']:
            analysis['correct_phonemes'] += 1
        else:
            if teacher['phoneme'] == '-':
                analysis['extra_phonemes'].append({
                    'position': i,
                    'phoneme': student['phoneme'],
                    'timestamp': f"{student['start']:.2f}-{student['end']:.2f}"
                })
            elif student['phoneme'] == '-':
                analysis['missing_phonemes'].append({
                    'position': i,
                    'phoneme': teacher['phoneme'],
                    'timestamp': f"{teacher['start']:.2f}-{teacher['end']:.2f}"
                })
            else:
                analysis['errors'].append({
                    'position': i,
                    'expected': teacher['phoneme'],
                    'pronounced': student['phoneme'],
                    'timestamp': f"{student['start']:.2f}-{student['end']:.2f}"
                })

        if teacher['phoneme'] != '-' and student['phoneme'] != '-':
            timing_diff = abs(teacher['end'] - teacher['start'] - 
                              (student['end'] - student['start']))
            if timing_diff > 0.1:
                analysis['timing_differences'].append({
                    'position': i,
                    'phoneme': teacher['phoneme'],
                    'teacher_duration': f"{teacher['end'] - teacher['start']:.2f}",
                    'student_duration': f"{student['end'] - student['start']:.2f}"
                })

    valid_phonemes = len(teacher_seq) - teacher_seq.count({'phoneme': '-', 'start': 0, 'end': 0})
    if valid_phonemes > 0:
        analysis['accuracy_percentage'] = (analysis['correct_phonemes'] / valid_phonemes) * 100

    return analysis

def generate_feedback(analysis: Dict) -> str:
    feedback = []
    feedback.append(f"Overall pronunciation accuracy: {analysis['accuracy_percentage']:.1f}%\n")

    if analysis['errors']:
        feedback.append("Pronunciation errors:")
        for error in analysis['errors']:
            feedback.append(f"- At {error['timestamp']}s: Expected '{error['expected']}' "
                            f"but heard '{error['pronounced']}'")

    if analysis['missing_phonemes']:
        feedback.append("\nMissing sounds:")
        for missing in analysis['missing_phonemes']:
            feedback.append(f"- Missing '{missing['phoneme']}' at {missing['timestamp']}s")

    if analysis['extra_phonemes']:
        feedback.append("\nExtra sounds:")
        for extra in analysis['extra_phonemes']:
            feedback.append(f"- Extra '{extra['phoneme']}' at {extra['timestamp']}s")

    if analysis['timing_differences']:
        feedback.append("\nTiming differences:")
        for timing in analysis['timing_differences']:
            feedback.append(f"- '{timing['phoneme']}': Teacher duration: {timing['teacher_duration']}s, "
                            f"Student duration: {timing['student_duration']}s")

    return "\n".join(feedback)

# Title
st.title("DSP Project")
LANGUAGE_ID = "eng"
language = st.selectbox("Choose Language", ["English", "Egyption Arabic", "Arabic"])

ReferenceUtterance = st.audio_input("Record a Reference Utterance")
if ReferenceUtterance is not None:
    reference_audio_path = AudioPreprocessor(ReferenceUtterance)
    st.write("Spectrogram for Reference Utterance")
    plot_spectrogram(reference_audio_path)

    TestUtterance = st.audio_input("Record a Test Utterance")
    if TestUtterance is not None:
        if language == "Arabic":
            LANGUAGE_ID = "arb"
        elif language == "English":
            LANGUAGE_ID = "eng"
        elif language == "Egyption Arabic":
            LANGUAGE_ID = "arz"

        universal_recognizer, english_recognizer = LoadRecognizer()
        test_audio_path = AudioPreprocessor(TestUtterance)
        st.write("Spectrogram for Test Utterance")
        plot_spectrogram(test_audio_path)

        try:
            if language == "English":
                teacher_phonemes = recognize_phonemes(english_recognizer, reference_audio_path, LANGUAGE_ID)
                student_phonemes = recognize_phonemes(english_recognizer, test_audio_path, LANGUAGE_ID)
            else:
                teacher_phonemes = recognize_phonemes(universal_recognizer, reference_audio_path, LANGUAGE_ID)
                student_phonemes = recognize_phonemes(universal_recognizer, test_audio_path, LANGUAGE_ID)

            if teacher_phonemes and student_phonemes:
                aligned_teacher, aligned_student = align_sequences(teacher_phonemes, student_phonemes)
                analysis = analyze_differences(aligned_teacher, aligned_student)
                feedback = generate_feedback(analysis)
                st.text(feedback)
        except Exception as e:
            st.error(f"Error during recognition: {e}")
