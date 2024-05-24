import streamlit as st
import os
from tortoise.models.classifier import AudioMiniEncoderWithClassifierHead
import librosa
import plotly.express as px
import torch
import torch.nn.functional as F
import torchaudio
import numpy as np
import io

# Function to load audio file
def load_audio(audiopath, sampling_rate=22050):
    if isinstance(audiopath, str):
        if audiopath.endswith('.mp3') or audiopath.endswith('.wav'):
            audio, lsr = librosa.load(audiopath, sr=sampling_rate)
            audio = torch.FloatTensor(audio)
        else:
            raise ValueError(f"Unsupported audio format provided: {audiopath[-4:]}")
    elif isinstance(audiopath, io.BytesIO):
        audio, lsr = torchaudio.load(audiopath)
        audio = audio[0]
        if lsr != sampling_rate:
            audio = torchaudio.functional.resample(audio, lsr, sampling_rate)
        if torch.any(audio > 1) or not torch.any(audio > 0):
            st.warning(f"Error with audio data. max={audio.max()} Min={audio.min()}")
        audio.clip_(0, 1)
    return audio.unsqueeze(0)

# Function for classifier
def classify_audio_clip(clip):
    classifier = AudioMiniEncoderWithClassifierHead(2, spec_dim=1, embedding_dim=512, depth=5, downsample_factor=4, resnet_blocks=2, attn_blocks=4, num_attn_heads=4, base_channels=32, dropout=0, kernel_size=5, distribute_zero_label=False)
    state_dict = torch.load('classifier.pth', map_location=torch.device('cpu'))
    classifier.load_state_dict(state_dict)
    clip = clip.cpu().unsqueeze(0)
    results = F.softmax(classifier(clip), dim=-1)
    return results[0][0]

# Streamlit app configuration
st.set_page_config(layout="wide")
st.title("AL-Generated Voice Detection")

# File uploader
uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3"])

# Display uploaded audio and analyze button
if uploaded_file is not None:
    if st.button("Analyze Audio"):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.info("Your results below")
            # Load and classify the audio file
            try:
                audio_clip = load_audio(uploaded_file)
                result = classify_audio_clip(audio_clip)
                st.info(f"Result probability: {result:.4f}")
                st.success(f"The uploaded audio is {result * 100:.2f}% likely to be AI generated.")
            except Exception as e:
                st.error(f"Error analyzing audio: {e}")
        with col2:
            st.info("Your uploaded audio is below")
            st.audio(uploaded_file)
            # Create a waveform plot
            try:
                fig = px.line()
                fig.add_scatter(x=list(range(len(audio_clip.squeeze()))), y=audio_clip.squeeze())
                fig.update_layout(title="Waveform Plot", xaxis_title="Time", yaxis_title="Amplitude")
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error plotting waveform: {e}")
        with col3:
            st.info("Disclaimer")
            st.warning("These classification or detection mechanisms are not always accurate. They should serve as a strong signal and not the ultimate decision makers.")