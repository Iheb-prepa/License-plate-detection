import gradio as gr
from src.main import main


def run_detection(video_path):
    output_video_path = main(video_path)  # main returns a Path object
    return str(output_video_path)


demo = gr.Interface(
    fn=run_detection,
    inputs=gr.Video(label="Upload Video"),
    outputs=gr.Video(label="Processed Video"),
    title="License Plate Detection & Tracking",
    description="Upload a video, and this app will detect, track, and read license plates!\n If the output video is hard to play, then please download it and play it locally."
)

demo.launch()