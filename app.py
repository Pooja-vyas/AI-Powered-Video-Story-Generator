import os
import tempfile
import streamlit as st
from moviepy.editor import VideoFileClip, concatenate_videoclips
import whisper
from ultralytics import YOLO

# Load models
@st.cache_resource
def load_whisper_model():
    return whisper.load_model("base")

@st.cache_resource
def load_yolo_model():
    return YOLO("yolov5s.pt")


whisper_model = load_whisper_model()
yolo_model = load_yolo_model()

# Streamlit UI
st.title("AI-Powered Video Story Generator")
st.text("Upload your videos and generate a story based on a prompt.")

uploaded_files = st.file_uploader("Upload Video Files", type=["mp4"], accept_multiple_files=True)

if uploaded_files:
    prompt = st.text_input("Enter a story prompt (e.g., 'beach', 'birthday party'):")

    if st.button("Generate Story"):
        all_matching_clips = []
        transcriptions = []

        for uploaded_file in uploaded_files:
            # Save uploaded video
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
                temp_video.write(uploaded_file.read())
                video_path = temp_video.name

            st.write(f"Processing video: {uploaded_file.name}...")

            # Extract audio and transcribe
            video = VideoFileClip(video_path)
            audio_path = video_path.replace(".mp4", ".wav")
            video.audio.write_audiofile(audio_path)

            result = whisper_model.transcribe(audio_path, verbose=False)
            transcription = result["text"]
            transcriptions.append(transcription)

            st.text(f"Transcription for {uploaded_file.name}: {transcription}")

            # Extract matching segments
            segments = result["segments"]
            for segment in segments:
                if prompt.lower() in segment["text"].lower():
                    start_time = segment["start"]
                    end_time = segment["end"]

                    # Extract matching clip
                    matching_clip = video.subclip(start_time, end_time)
                    all_matching_clips.append(matching_clip)

        # Combine clips
        if all_matching_clips:
            combined_clip = concatenate_videoclips(all_matching_clips, method="compose")

            # Save combined video
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_combined_video:
                combined_clip.write_videofile(temp_combined_video.name, codec="libx264", audio_codec="aac")
                combined_video_path = temp_combined_video.name

            st.success("Story created successfully!")
            st.video(combined_video_path)

            # Provide download link
            with open(combined_video_path, "rb") as file:
                st.download_button(
                    label="Download Combined Story",
                    data=file,
                    file_name="story.mp4",
                    mime="video/mp4"
                )
        else:
            st.warning("No matching segments found for the given prompt.")
