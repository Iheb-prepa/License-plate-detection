import os
import cv2
import numpy as np
from sort.sort import Sort
from ultralytics import YOLO
from pathlib import Path
import yaml
import construct_final_video 
from paddleocr import PaddleOCR
import logging
from utils import *


logging.getLogger('ppocr').setLevel(logging.WARNING)  # or logging.ERROR

# Mapping dictionaries for character conversion
CHAR_TO_INT_DICT = {'O': '0',
                    'I': '1',
                    'J': '3',
                    'A': '4',
                    'G': '6',
                    'S': '5'}

INT_TO_CHAR_DICT = {'0': 'O',
                    '1': 'I',
                    '3': 'J',
                    '4': 'A',
                    '6': 'G',
                    '5': 'S'}


def initialize_models(lp_model_weights_path: "Path", vehicle_model_weights_path: "Path") -> tuple:
    """Initialize the models for vehicle and license plate detection.
    This function initializes the PaddleOCR for OCR, YOLOv8 for vehicle detection,
    YOLOv8 for license plate detection, and SORT for tracking.
    It returns the initialized models.

    Args:
        lp_model_weights_path (Path): Path to the weights of the trained YOLO license plate detection model.
        vehicle_model_weights_path (Path): Path to the weights of the YOLOv8s vehicle detection model.

    Returns:
        tuple: initialized models (PaddleOCR, YOLOv8 for vehicle detection, YOLOv8 for license plate detection, SORT).
    """    

    # Initialize the OCR reader
    paddle_ocr_reader = PaddleOCR(use_angle_cls=True, lang='en')    
    # reader = easyocr.Reader(['en']) # uncomment and change variable name if you want to use easyocr instead

    # Load YOLOv8 model for vehicle detection
    model = YOLO(vehicle_model_weights_path)  

    # Initialize YOLO license plate detector
    license_plate_model = YOLO(lp_model_weights_path)

    # Initialize SORT tracker
    tracker = Sort()
    return paddle_ocr_reader, model, license_plate_model, tracker


def perform_vehicle_detection(detections: list, results, model, VEHICLE_CLASSES: list) -> list:
    """Uses YOLOv8 model to detect vehicles in the frame.

    Args:
        detections (list): a (given empty) list to store the detected vehicles' bounding boxes and confidence scores. 
        results (_type_): detection results from the YOLOv8 model.
        model (instance of class ): the model used for vehicle detection.
        VEHICLE_CLASSES (list): Supported vehicle classes for detection

    Returns:
        list: a list of lists of the coordinates of the detected vehicles bboxes.
    """    
    for result in results[0].boxes:  # Access boxes directly from results
        x1, y1, x2, y2 = map(int, result.xyxy[0].cpu().numpy())  # Convert to CPU and NumPy
        conf = float(result.conf[0].cpu().numpy())  # Confidence score to float
        cls = int(result.cls[0].cpu().numpy())  # Class index to integer
        class_name = model.names[cls]

        if class_name in VEHICLE_CLASSES and conf > 0.4:  # Confidence threshold
            detections.append([x1, y1, x2, y2, conf])  # Add detection with CPU-based data

    # Update SORT tracker with detections
    detections = np.array(detections) if detections else np.empty((0, 5))
    return detections


def save_vehicles_coordinates(tracked_objects, car_id_and_coordinates: dict, frame_number: int, csv_data_dict) -> None:
    """Save the coordinates of the detected vehicles in a dictionary that'll be used to update the CSV data dictionary.

    Args:
        tracked_objects (_type_): contains the coordinates of the detected vehicles.
        car_id_and_coordinates (dict): a dictionary to store the coordinates of the detected vehicles.
        frame_number (int): the current frame number.
        csv_data_dict (_type_): a dictionary to store the CSV data.
    """    
    for x1, y1, x2, y2, track_id in tracked_objects:
        x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))  # Ensure coordinates are integers
        car_id_and_coordinates[int(track_id)] = (x1, y1, x2, y2) # Putting the car ID on the corresponding car bbox
        bbox = [x1, y1, x2, y2]
        csv_data_dict[(frame_number, int(track_id))] = [frame_number, int(track_id), str(bbox), '', '', '']


def run_ocr_on_license_plate_and_save_to_dict(country: str, paddle_ocr_reader: PaddleOCR, cropped_region: np.ndarray, car_id: int,
                                             car_and_lp: dict, csv_data_dict: dict, frame_number: int, lp_bbox: list, logs: list) -> None:
    """Runs OCR on the cropped license plate region from the frame, performs eventual reading corrections 
      and saves the reading to the dictionary. (Technically, the dictionay stores the reading and the number of occurences of this reading
      so that later we can correct the reading by taking the most frequent one.)

    Args:
        country (str): specifies the country of the license plate 'fr' for french or 'eng' for english.
        paddle_ocr_reader (PaddleOCR): instance of the PaddleOCR class.
        cropped_region (np.ndarray): cropped image of the license plate.
        car_id (int): ID of the detected vehicle.
        car_and_lp (dict): dictionary to store license plate readings and their occurrences.
        csv_data_dict (dict): dictionary to store CSV data.
        frame_number (int): the current frame number.
        lp_bbox (list): coordinates of the license plate bounding box.
        logs (list): list to store logs.
    """    
    paddle_result = paddle_ocr_reader.ocr(cropped_region, cls=True)
    if paddle_result[0] is not None:
        for line in paddle_result[0]:  # 0 because PaddleOCR supports batch of images
            (lp_id, confidence) = line[1]  # text and confidence
            lp_id = correction_license_plate_country(country, lp_id, CHAR_TO_INT_DICT, INT_TO_CHAR_DICT, logs)
            if lp_id is None:
                continue
            # car_id is a dict[dict], {car_id: {lp_id_1:occurences_lp1, lp_id_2: occurences_lp2},... } 
            # A car should have one license plate number but the ocr may give different readings depending on the frame for the 
            # same object, so we construct this dictionary and use it later to correct the readings.
            if car_id is not None:
                if car_id not in car_and_lp.keys():
                    car_and_lp[car_id] = {lp_id:1}
                else:
                    if lp_id in car_and_lp[car_id].keys():
                        car_and_lp[car_id][lp_id] += 1
                    else:
                        car_and_lp[car_id][lp_id] = 1
                update_lp_data(csv_data_dict, frame_number, car_id, lp_id, [lp_bbox], confidence)


def assign_license_plate_to_vehicle_then_read_lp(country: str, box: np.ndarray, frame: np.ndarray, car_id_and_coordinates: dict, 
                                                 csv_data_dict: dict, frame_number: int, 
                                                paddle_ocr_reader: PaddleOCR, logs: list, car_and_lp: dict) -> None:
    """Crops the license plate region from the frame, assigns it to the corresponding vehicle ID,
    and runs OCR to read the license plate. The results are saved in the csv_data_dict.

    Args:
        country (str): specifies the country of the license plate 'fr' for french or 'eng' for english.
        box (np.ndarray): bounding box coordinates of the detected license plate.
        frame (np.ndarray): the current frame of the video.
        car_id_and_coordinates (dict): a dictionary to store the coordinates of the detected vehicles.
        csv_data_dict (dict): a dictionary to store the CSV data.
        frame_number (int): the current frame number.
        paddle_ocr_reader (PaddleOCR): instance of the PaddleOCR class.
        logs (list): a list to store logs.
        car_and_lp (dict): dictionary to store license plate readings and their occurrences.
    """    
    
    lp_x1, lp_y1, lp_x2, lp_y2 = map(int, box.xyxy[0])  # Get license plate box coordinates in the frame
    # Check if coordinates are within the frame dimensions
    if lp_x1 >= 0 and lp_y1 >= 0 and lp_x2 <= frame.shape[1] and lp_y2 <= frame.shape[0]:
        # Crop the detected region
        cropped_region = frame[lp_y1:lp_y2, lp_x1:lp_x2]
        lp_bbox = lp_x1, lp_y1, lp_x2, lp_y2
        car_id = find_car_id(lp_bbox, car_id_and_coordinates)
        # Update the csv_data_dict with the lp bbox 
        update_lp_data(csv_data_dict, frame_number, car_id, '', [lp_bbox], '')

        if cropped_region.size > 0:  # Ensure the cropped region is not empty                        
            run_ocr_on_license_plate_and_save_to_dict(country, paddle_ocr_reader, cropped_region, car_id,
                                                       car_and_lp, csv_data_dict, frame_number, lp_bbox, logs)

        else:
            print("Cropped region is empty.")
    else:
        print(f"Invalid crop coordinates: ({lp_x1}, {lp_y1}), ({lp_x2}, {lp_y2}) - skipping OCR.")


def main(VIDEO_PATH: "Path", country='fr') -> "Path":
    """Main function to process the video, detect vehicles and license plates, and save the results.
    This function initializes the models, processes each frame of the video, performs vehicle detection, performs license plate detection,
    assigns license plates to vehicles, and saves the results in a CSV file and a YAML file. 
    Uses the Yaml file to find the most likely license plate number for each vehicle and corrects the CSV file to have the (hopefully) correct license plate number for each vehicle.
    It then generates a final video with the detected vehicles and license plates bounding boxes and lp numbers using the corrected CSV file and the initial video.

    Args:
        VIDEO_PATH (Path): Path to the input video file.
        country (str, optional): specifies the country of the license plate 'fr' for french or 'eng' for english. Defaults to 'fr'.

    Returns:
        Path: Path to the final output video file.
    """    
    logs = []

    BASE_DIR = Path(__file__).resolve().parent.parent  # Go up to the project root
    lp_model_weights_path = BASE_DIR / "yolo_weights" / "weights.pt" 
    vehicle_model_weights_path = BASE_DIR / "yolo_weights" / "yolov8s.pt"  # Path to the YOLOv8 vehicle detection model weights

    # Define vehicle classes based on YOLO model
    VEHICLE_CLASSES = ['car', 'truck', 'bus', 'motorcycle']
    VIDEO_NAME = os.path.splitext(os.path.basename(VIDEO_PATH))[0]  
    YAML_FILE_PATH = BASE_DIR / 'output' / f"output_{VIDEO_NAME}.yaml"
    OUTPUT_VIDEO_PATH = BASE_DIR / 'output' / f"tracked_output_{VIDEO_NAME}.mp4"

    car_and_lp = {}
    csv_data_dict = {}

    # Initialize the csv file
    csv_output_path = BASE_DIR / 'output' / f'detections_{VIDEO_NAME}.csv'
    CSV_HEADER = ['frame_number', 'car_id', 'car_bbox', 'lp_id', 'lp_bbox', 'lp_confidence']

    # Open source video
    cap = cv2.VideoCapture(VIDEO_PATH)

    paddle_ocr_reader, model, license_plate_model, tracker = initialize_models(lp_model_weights_path, vehicle_model_weights_path)

    frame_number = 1
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # YOLOv8 model inference
        results = model(frame, verbose=False)

        # Filter results to include only specified vehicle classes
        detections = []
        detections = perform_vehicle_detection(detections, results, model, VEHICLE_CLASSES)

        tracked_objects = tracker.update(detections)

        car_id_and_coordinates = {}
        
        # Save bounding boxes coordinates and IDs of vehicles in csv_data_dict (dictionary)
        save_vehicles_coordinates(tracked_objects, car_id_and_coordinates, frame_number, csv_data_dict)

        # Detect license plates within the frame using the YOLO license plate detection model
        license_plate_results = license_plate_model.predict(frame, conf=0.5, save=False, verbose=False)

        for lp_result in license_plate_results:
            for box in lp_result.boxes:
                assign_license_plate_to_vehicle_then_read_lp(country, box, frame, car_id_and_coordinates, csv_data_dict, frame_number, 
                                                    paddle_ocr_reader, logs, car_and_lp)
                
        print("treated frame: ", frame_number)
        frame_number += 1

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

    # Save the license plates readings to the YAML file 
    with open(YAML_FILE_PATH, 'w') as file:
        yaml.dump(car_and_lp, file, default_flow_style=False)
    print(f"Dictionary written to {YAML_FILE_PATH}")

    # Write the CSV file with the detected objects information (csv with the columns: [frame_number,car_id,car_bbox,lp_id,lp_bbox,lp_confidence])
    write_csv(CSV_HEADER, csv_data_dict, csv_output_path)

    # Correct the CSV file with the most correct license plate number for each vehicle
    generate_corrected_csv(YAML_FILE_PATH, csv_output_path)

    # Generate the final video 
    construct_final_video.main(csv_output_path, VIDEO_PATH, OUTPUT_VIDEO_PATH)

    return OUTPUT_VIDEO_PATH
    

if __name__ == '__main__':
    VIDEO_NAME = 'test7'
    
    BASE_DIR = Path(__file__).resolve().parent.parent  # Go up to the project root
    VIDEO_PATH = BASE_DIR / "test_videos" / f"{VIDEO_NAME}.mp4"

    if VIDEO_NAME == 'test1':
        country = 'eng'
    elif VIDEO_NAME == 'test5' or VIDEO_NAME == 'test6' or VIDEO_NAME == 'test7' or VIDEO_NAME == 'test7-bis' or VIDEO_NAME == 'test8' or VIDEO_NAME == 'test8-rgb-crop':
        country = 'fr'
    else:
        print("EXITING: please select a country for the license plates.")
        exit(1)
    main(VIDEO_PATH, country)