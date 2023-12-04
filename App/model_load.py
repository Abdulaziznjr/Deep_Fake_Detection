import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
import librosa
import matplotlib.pyplot as plt
import librosa.display

# Constants
SAMPLE_RATE = 44100
N_MELS = 128
max_time_steps = 109

# Load your audio classification model
model = load_model("audio_classifier18.h5")

st.set_page_config(
    page_title="Deep Fake Detection",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Function to preprocess audio
def preprocess_audio(audio, target_sr=44100, target_shape=(128, 109)):
    mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=SAMPLE_RATE, n_mels=N_MELS)
    mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)

    # Ensure all spectrograms have the same width (time steps)
    if mel_spectrogram.shape[1] < max_time_steps:
        mel_spectrogram = np.pad(mel_spectrogram, ((0, 0), (0, max_time_steps - mel_spectrogram.shape[1])), mode='constant')
    else:
        mel_spectrogram = mel_spectrogram[:, :max_time_steps]

    return mel_spectrogram

# Function to make predictions
def predict(processed_data):
    # Make sure there are no non-finite values in the processed_data
    if not np.all(np.isfinite(processed_data)):
        st.error("Non-finite values detected in the processed audio data. Unable to make predictions.")
        return None

    # Make predictions using your model
    prediction = model.predict(processed_data.reshape(1, 128, 109, 1))

    # Assuming binary classification with softmax output
    class_labels = ["IT IS A FAKE VOICE", "IT IS A REAL VOICE"]
    predicted_class = class_labels[int(prediction[0][0] > 0.5)]

    return predicted_class

# Streamlit app
import base64
import plotly.express as px

# Get base64 encoded image
def get_img_as_base64(file):
    with open(file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

img = get_img_as_base64("c:/Users/3bdal\Downloads/frequency.jpeg")

# Page background styling
page_bg_img = f"""
<style>
[data-testid="stAppViewContainer"] > .main {{
    background-image: url("https://img.freepik.com/free-vector/sinuous-background_23-2148487911.jpg?size=626&ext=jpg&ga=GA1.1.2008272138.1697155200&semt=ais");
    background-size: 100%;
    background-position: top left;
    background-repeat: repeat;
    background-attachment: local;
}}

[data-testid="stSidebar"] > div:first-child {{
    background-image: url("data:image/png;base64,{img}");
    background-position: center; 
    background-repeat: no-repeat;
    background-attachment: fixed;
}}

[data-testid="stHeader"] {{
    background: rgba(0,0,0,0);
}}

[data-testid="stToolbar"] {{
    right: 2rem;
}}
</style>
"""

st.markdown(page_bg_img, unsafe_allow_html=True)
st.title("Deep Fake Detection")
st.title("Upload Audio")

# Option to upload an audio file
uploaded_file = st.file_uploader("Choose an audio file", type=["wav", "mp3", "flac"])

if uploaded_file:
    # Display uploaded file details
    st.audio(uploaded_file, format='audio/wav')
    
    # Preprocess the uploaded audio file
    audio_file = librosa.load(uploaded_file, sr=SAMPLE_RATE)[0]
    processed_data_file = preprocess_audio(audio_file)

    if processed_data_file is not None:
        # Display file details
        st.write("File details:")
        st.write(f"Sample Rate: {SAMPLE_RATE} Hz")
        st.write(f"Duration: {len(audio_file) / SAMPLE_RATE:.2f} seconds")

        # Plot the waveform and spectrogram side by side
        st.subheader("Waveform and Spectrogram:")
        col1_file, col2_file = st.columns(2)

        with col1_file:
            # Plot the waveform
            fig_waveform_file, ax_waveform_file = plt.subplots(figsize=(10, 4))
            librosa.display.waveshow(audio_file, sr=SAMPLE_RATE, ax=ax_waveform_file)
            st.pyplot(fig_waveform_file)

        with col2_file:
            # Plot the spectrogram
            fig_spec_file, ax_spec_file = plt.subplots(figsize=(10, 4))
            img_file = librosa.display.specshow(processed_data_file[:, :], sr=SAMPLE_RATE, x_axis='time', y_axis='mel', ax=ax_spec_file)
            colorbar_file = plt.colorbar(format='%+2.0f dB', ax=ax_spec_file, mappable=img_file)
            st.pyplot(fig_spec_file)

        # Add a Predict button for uploaded file
        if st.button("Predict (Uploaded File)"):
            # Make predictions
            predicted_class_file = predict(processed_data_file)

            # Display predictions with colored words
            if predicted_class_file is not None:
                colored_prediction = f"<p style='font-size:50px;color:{'red' if predicted_class_file == 'IT IS A FAKE VOICE' else 'green'}'>{predicted_class_file}</p>"
                st.markdown(colored_prediction, unsafe_allow_html=True)
