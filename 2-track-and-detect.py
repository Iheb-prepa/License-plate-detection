import re
import cv2
import numpy as np
from sort.sort import Sort
from ultralytics import YOLO
# import easyocr
import yaml
import csv
import postprocess_results 
import construct_final_video 
from paddleocr import PaddleOCR



def resize_cropped_image(cropped_image, scale_factor=2.0):
    # Resize the image using the specified scale factor
    new_width = int(cropped_image.shape[1] * scale_factor)
    new_height = int(cropped_image.shape[0] * scale_factor)
    return cv2.resize(cropped_image, (new_width, new_height))


def is_valid_fr(corrected_text):
    detected_text = detected_text.upper()
    if len(detected_text) != 7:
        return False
    if not detected_text[:2].isalpha():
        return False
    if not detected_text[2:5].isdigit():
        return False
    if not detected_text[5:7].isalpha():
        return False
    return True
        

def correction_french(detected_text, CHAR_TO_INT_DICT, INT_TO_CHAR_DICT, logs):
    n = len(detected_text)
    print(detected_text, n)
    if (n < 7):
        return None
    
    pattern = r'[^\w]*([A-Za-z013456]{2}[^\w]*[0-9AGSOIJ]{3}[^\w]*[A-Za-z013456]{2})[^\w]*'
    match = re.search(pattern, detected_text)
    if not match:
        return None
    result_without_special_chars = re.sub(r'[^\w]', '', match.group(1))
    print(result_without_special_chars)
    corrected_text = ''
    for i in range(2): # correcting first two letters
        if result_without_special_chars[i].isalpha():
            corrected_text += result_without_special_chars[i].upper()
        elif result_without_special_chars[i] in INT_TO_CHAR_DICT.keys():
            corrected_text += INT_TO_CHAR_DICT[result_without_special_chars[i]]
        else:
            return None
        
    
    for i in range(2,5): # correcting the two digits
        if result_without_special_chars[i].isdigit():
            corrected_text += result_without_special_chars[i]
        elif result_without_special_chars[i] in CHAR_TO_INT_DICT:
            corrected_text += CHAR_TO_INT_DICT[result_without_special_chars[i]]
        else:
            return None


    for i in range(5, 7): # correcting the last 3 letters
        if result_without_special_chars[i].isalpha():
            corrected_text += result_without_special_chars[i].upper()
        elif result_without_special_chars[i] in INT_TO_CHAR_DICT.keys():
            corrected_text += INT_TO_CHAR_DICT[result_without_special_chars[i]]
        else:
            return None
        
    logs.append(corrected_text)
    return corrected_text
    

def correction_british(detected_text, CHAR_TO_INT_DICT, INT_TO_CHAR_DICT):
    n = len(detected_text)
    if (n < 7):
        return None
    
    corrected_text = ''
    for i in range(2): # correcting first two letters
        if detected_text[i].isalpha():
            corrected_text += detected_text[i].upper()
        elif detected_text[i] in INT_TO_CHAR_DICT.keys():
            corrected_text += INT_TO_CHAR_DICT[detected_text[i]]
        else:
            return None
    
    for i in range(2,4): # correcting the two digits
        if detected_text[i].isdigit():
            corrected_text += detected_text[i]
        elif detected_text[i] in CHAR_TO_INT_DICT:
            corrected_text += CHAR_TO_INT_DICT[detected_text[i]]
        else:
            return None
        
    for i in range(n-3, n): # correcting the last 3 letters
        if detected_text[i].isalpha():
            corrected_text += detected_text[i].upper()
        elif detected_text[i] in INT_TO_CHAR_DICT.keys():
            corrected_text += INT_TO_CHAR_DICT[detected_text[i]]
        else:
            return None
    
    return corrected_text


def correction_license_plate_country(country, detected_text, CHAR_TO_INT_DICT, INT_TO_CHAR_DICT, logs):
    if country == 'eng':
        return correction_british(detected_text, CHAR_TO_INT_DICT, INT_TO_CHAR_DICT)
    elif country == 'fr':
        return correction_french(detected_text, CHAR_TO_INT_DICT, INT_TO_CHAR_DICT, logs)


def is_contained(bbox1, bbox2):
    """
    Check if bbox1 is fully contained within bbox2.

    Parameters:
    bbox1 (tuple): Coordinates of the first bounding box as (x1, y1, x2, y2).
    bbox2 (tuple): Coordinates of the second bounding box as (x1, y1, x2, y2).

    Returns:
    bool: True if bbox1 is contained within bbox2, False otherwise.
    """
    x1_1, y1_1, x2_1, y2_1 = bbox1
    x1_2, y1_2, x2_2, y2_2 = bbox2

    # Check if bbox1 is within bbox2
    return (x1_2 <= x1_1 <= x2_2 and
            y1_2 <= y1_1 <= y2_2 and
            x1_2 <= x2_1 <= x2_2 and
            y1_2 <= y2_1 <= y2_2)


def find_car_id(lp_coordinates, car_id_and_coordinates):
    """
    For a given license plate (lp_coordiantes), find the corresponding car (car_id) 
    among the detected cars (car_id_and_coordinates). 

    Parameters:
    lp_coordinates (tuple): Coordinates of the license plate.
    car_id_and_coordinates (dictionary): Car ids with their bboxes.

    Returns:
    bool: Car id of the given license plate, None otherwise.
    """
    for car_id, car_coordinates in car_id_and_coordinates.items():
        if is_contained(lp_coordinates, car_coordinates):
            return car_id
    return None


def update_lp_data(csv_data, frame_number, car_id, lp_id, lp_bbox, lp_confidence):
    # Filling the dictionary csv_data that has the columns: ['frame_number', 'car_id', 'car_bbox', 'lp_id', 'lp_bbox', 'lp_confidence']
    if (frame_number, car_id) in csv_data:
        row = csv_data[(frame_number, car_id)] # car_bbox should already be in csv_data
        row[3] = str(lp_id)
        row[4] = str(lp_bbox)  # Update lp_bbox
        row[5] = lp_confidence  # Update lp_confidence
        csv_data[(frame_number, car_id)] = row
        print("Got update lp data,", csv_data[(frame_number, car_id)])


def generate_corrected_csv(YAML_FILE_PATH, csv_output_path):
    print("Opening yaml file")
    with open(YAML_FILE_PATH, "r") as file:
        data = yaml.safe_load(file)
    # Extract the most occurring (most likely) license plate value for each detected license plate
    license_plate_ids_dict = postprocess_results.correct_all_license_plates(data)
    print(license_plate_ids_dict)

    # correct the csv file
    with open(csv_output_path, mode='r', newline='') as input_file:
        csv_reader = csv.DictReader(input_file)
        rows = list(csv_reader)  # Read all rows into memory

    # Modify the rows
    for row in rows:
        car_id = int(row['car_id'])  # Ensure car_id is an integer
        if car_id in license_plate_ids_dict:
            # print("YES SIR")
            row['lp_id'] = license_plate_ids_dict[car_id]  # Update lp_id

    # Overwrite the existing CSV
    with open(csv_output_path, mode='w', newline='') as output_file:
        csv_writer = csv.DictWriter(output_file, fieldnames=csv_reader.fieldnames)
        csv_writer.writeheader()  # Write the header
        csv_writer.writerows(rows)  # Write the modified rows

    print(f"CSV updated successfully at {csv_output_path}")


def write_csv(header, data, output_path):
    """
    Writes a CSV file with the given header and data.

    Parameters:
        header (list): List of column names for the CSV file.
        data (dict): Dictionary where keys are tuples (frame_number, track_id)
                     and values are lists containing row data.
        output_path (str): Path where the CSV file will be saved.
    """
    with open(output_path, mode='w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        # Write the header to the CSV file
        writer.writerow(header)
        
        # Write the rows to the CSV file
        for key, values in data.items():
            writer.writerow(values)


def main(VIDEO_NAME, country):
    logs = []
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

    # Initialize the EasyOCR reader
    # reader = easyocr.Reader(['en'])

    paddle_ocr_reader = PaddleOCR(use_angle_cls=True, lang='en')    

    # Load YOLOv8 model
    model = YOLO('yolov8s.pt')  # Replace 'yolov8s.pt' with your specific YOLOv8 model if needed

    license_plate_model = YOLO("runs/detect/train4/weights/best.pt")

    # Define vehicle classes based on YOLO model
    VEHICLE_CLASSES = ['car', 'truck', 'bus', 'motorcycle']
    VIDEO_PATH = f'test_videos/{VIDEO_NAME}.mp4'
    YAML_FILE_PATH = f'output_{VIDEO_NAME}.yaml'
    OUTPUT_VIDEO_PATH = f'tracked_output_{VIDEO_NAME}.mp4'
    CODEC = 'mp4v'

    # Initialize SORT tracker
    tracker = Sort()

    # Open video source
    cap = cv2.VideoCapture(VIDEO_PATH)

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Define the codec and create VideoWriter object to save the result
    fourcc = cv2.VideoWriter_fourcc(*CODEC)  # Codec
    out = cv2.VideoWriter(f"INTERMEDIATE_VIDEO_{VIDEO_NAME}.mp4", cv2.VideoWriter_fourcc(*'avc1'), fps, (frame_width, frame_height))


    # Font and bounding box settings
    FONT_SCALE = 1.5
    BOX_THICKNESS = 10
    TEXT_THICKNESS = 5

    car_and_lp = {}
    csv_data = {}

    # Initialize the csv file
    csv_output_path = f'detections_{VIDEO_NAME}.csv'
    CSV_HEADER = ['frame_number', 'car_id', 'car_bbox', 'lp_id', 'lp_bbox', 'lp_confidence']
    

    frame_number = 1
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("BREAKING !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            break

        # YOLOv8 model inference
        results = model(frame)

        # Filter results to include only specified vehicle classes
        detections = []
        for result in results[0].boxes:  # Access boxes directly from results
            x1, y1, x2, y2 = map(int, result.xyxy[0].cpu().numpy())  # Convert to CPU and NumPy
            conf = float(result.conf[0].cpu().numpy())  # Confidence score to float
            cls = int(result.cls[0].cpu().numpy())  # Class index to integer
            class_name = model.names[cls]

            if class_name in VEHICLE_CLASSES and conf > 0.4:  # Confidence threshold
                detections.append([x1, y1, x2, y2, conf])  # Add detection with CPU-based data

        # Update SORT tracker with detections
        # if detections:
        #     detections = np.array(detections)  # Convert to NumPy array if not empty
        detections = np.array(detections) if detections else np.empty((0, 5))
        # print(detections)
        tracked_objects = tracker.update(detections)

        car_id_and_coordinates = {}
        
        # Draw bounding boxes and IDs
        for x1, y1, x2, y2, track_id in tracked_objects:
            x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))  # Ensure coordinates are integers
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), BOX_THICKNESS)
            cv2.putText(frame, f'ID {int(track_id)}', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, (255, 0, 0), TEXT_THICKNESS)
            car_id_and_coordinates[int(track_id)] = (x1, y1, x2, y2)

            bbox = [x1, y1, x2, y2]
            # csv_writer.writerow([frame_number, int(track_id), str(bbox), '', '', ''])
            csv_data[(frame_number, int(track_id))] = [frame_number, int(track_id), str(bbox), '', '', '']

        license_plate_results = license_plate_model.predict(frame, conf=0.25, save=False)
        # Draw bounding boxes on the frame
        for lp_result in license_plate_results:
            for box in lp_result.boxes:
                lp_x1, lp_y1, lp_x2, lp_y2 = map(int, box.xyxy[0])  # Get box coordinates
                label = license_plate_model.names[int(box.cls)]
                cv2.rectangle(frame, (lp_x1, lp_y1), (lp_x2, lp_y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (lp_x1, lp_y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                # Check if coordinates are within the frame dimensions
                if lp_x1 >= 0 and lp_y1 >= 0 and lp_x2 <= frame.shape[1] and lp_y2 <= frame.shape[0]:
                    # Crop the detected region and run OCR
                    cropped_region = frame[lp_y1:lp_y2, lp_x1:lp_x2]
                    
                    lp_crop_gray = cv2.cvtColor(cropped_region, cv2.COLOR_BGR2GRAY)
                    _, lp_crop_threshold = cv2.threshold(lp_crop_gray, 64, 255, cv2.THRESH_BINARY_INV)
                    
                    lp_bbox = lp_x1, lp_y1, lp_x2, lp_y2
                    car_id = find_car_id(lp_bbox, car_id_and_coordinates)
                    update_lp_data(csv_data, frame_number, car_id, '', [lp_bbox], '')

                    if lp_crop_threshold.size > 0:  # Ensure the cropped region is not empty                        
                        # lp_crop_threshold_rgb = cv2.cvtColor(lp_crop_threshold, cv2.COLOR_GRAY2BGR)

                        # cv2.imshow('Cropped License Plate rgb', lp_crop_threshold_rgb)
                        # cv2.waitKey(0)
                        # cv2.destroyAllWindows() 

                        # ocr_result = reader.readtext(lp_crop_threshold_rgb)

                        paddle_result = paddle_ocr_reader.ocr(cropped_region, cls=True)

                        a = 0
                        if paddle_result[0] is not None:
                            for line in paddle_result[0]:  # 0 because PaddleOCR supports batch of images
                                ocr_bbox = line[0]  # bounding box points
                                (lp_id, confidence) = line[1]  # text and confidence
                            # Display OCR results on the frame
                            # for (ocr_bbox, lp_id, confidence) in ocr_result:
                                a+=1
                                lp_id = correction_license_plate_country(country, lp_id, CHAR_TO_INT_DICT, INT_TO_CHAR_DICT, logs)
                                if lp_id is None:
                                    continue
                                cv2.putText(frame, lp_id, (lp_x1, lp_y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
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
                                    update_lp_data(csv_data, frame_number, car_id, lp_id, [lp_bbox], confidence)

                    else:
                        print("Cropped region is empty.")
                else:
                    print(f"Invalid crop coordinates: ({lp_x1}, {lp_y1}), ({lp_x2}, {lp_y2}) - skipping OCR.")
                
        # Write the frame to the output video file
        out.write(frame)
        print("treated frame: ", frame_number)
        frame_number += 1

    # Release resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()


    # Write dictionary to YAML file (THIS BLOCK WAS IN THE LOOP)
    with open(YAML_FILE_PATH, 'w') as file:
        yaml.dump(car_and_lp, file, default_flow_style=False)
    print(f"Dictionary written to {YAML_FILE_PATH}")

    write_csv(CSV_HEADER, csv_data, csv_output_path)

    # csv_file.close()

    generate_corrected_csv(YAML_FILE_PATH, csv_output_path)

    construct_final_video.main(csv_output_path, VIDEO_PATH, OUTPUT_VIDEO_PATH)
    print(logs)
    

if __name__ == '__main__':
    VIDEO_NAME = 'test7'


    if VIDEO_NAME == 'test1':
        country = 'eng'
    elif VIDEO_NAME == 'test5' or VIDEO_NAME == 'test6' or VIDEO_NAME == 'test7' or VIDEO_NAME == 'test7-bis' or VIDEO_NAME == 'test8' or VIDEO_NAME == 'test8-rgb-crop':
        country = 'fr'
    else:
        print("EXITING: please select a country for the license plates.")
        exit(1)
    main(VIDEO_NAME, country)