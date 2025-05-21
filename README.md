# 🚗 License Plate and Vehicle Tracker with OCR

This project is a complete pipeline for detecting vehicles and license plates in video streams, tracking them across frames, and reading license plate numbers using Optical Character Recognition (OCR) for french and british license plates.

It combines state-of-the-art models and tracking algorithms:

-Two YOLOv8 models for object detection: one for detecting vehicles (pretrained) and the other is for license plates detection (fine-tuned on images of license plates).

-PaddleOCR for text recognition.

-SORT for object tracking.

# 📹 Example Output
[![Demo Video](https://img.youtube.com/vi/PwBZghOF5JkD/0.jpg)](https://www.youtube.com/watch?v=PwBZghOF5Jk)


# 🔧 Installation
1. Clone the repository
   
```
git clone https://github.com/iheb-ennine/license-plate-tracker-ocr.git
cd license-plate-tracker-ocr
```

2. Set up environment
   
It is recommended to use a Conda environment:

```
conda create -n plate-tracker-env python=3.10
conda activate plate-tracker-env
```

3. Install dependencies
```
pip install -r requirements.txt
conda install -c conda-forge opencv
```
4. Test your installation
```
import torch
print(torch.cuda.is_available())
# You should get true
```



# 🧪 Demo
You are provided with a test video (check under test_videos/test_video.mp4) that you can use for the demo, or you can choose your own video.

## First method (works only for french license plate format):
```
# Under license-plate-tracker/ 
python app.py
```
You should have a link in terminal to a local huggingface gradio interface, open the url, upload your video and launch the processing.

## Second method:

Place your input video under the folder test_videos/ (just like test_videos/test_video.mp4) go to main.py, add the name of your video and the country format ('fr or 'eng') in the variable country, then run:
```
# Under license-plate-tracker/ 
python src/main.py
```

# 🧾 Tips
-You can change the confidence level for detecting license plates by changing license_plate_model.predict() argument. Increasing the confidence would result in less but more accurate detections, decreasing the confidence would result in more license plates detections but also more false positives (i.e. other objects detected as license plates). 

-If you get an FFMPEG error when launching the program and at the end of the execution (the error won't stop the execution), consider reinstalling opencv with conda-forge

-This is an offline solution not a real-time one.

