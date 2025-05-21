import cv2
import ast  # To parse the bbox strings as lists
import csv
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


def main(csv_path: "Path", video_path: "Path", output_video_path: "Path") -> None:
    """
    Constructs the final video with the bounding boxes and license plates numbers.
    The function simply iterates through the frames of the original video, adds the bboxes and lp numbers from the csv and 
    saves the output into a new video.

    Args:
        csv_path (Path): Path of the (already generated) csv file that contains: frame_number, car_id, car_bbox, lp_id, lp_bbox, lp_confidence
        video_path (Path): Path of the original video
        output_video_path (Path): Path of the video to be generated

    Raises:
        ValueError: in case the original video cannot be opened
    """    
    # Parameters for drawing
    CAR_COLOR = (255, 0, 0)  # Blue for car bounding boxes
    LP_COLOR = (0, 255, 0)   # Green for license plate bounding boxes
    CAR_BOX_THICKNESS = 5
    FONT = cv2.FONT_HERSHEY_SIMPLEX
    FONT_SCALE = 1.5
    FONT_THICKNESS = 5
    LP_FONT_SCALE = 1.5
    LP_FONT_THICKNESS = 4

    # Load the CSV data into memory
    detections = {}
    with open(csv_path, mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        for row in csv_reader:
            frame_number = int(row['frame_number'])
            car_bbox = ast.literal_eval(row['car_bbox'])  # Convert bbox string to list
            lp_bbox = ast.literal_eval(row['lp_bbox']) if row['lp_bbox'] else None
            car_id = int(row['car_id'])
            lp_id = row['lp_id']
            detections.setdefault(frame_number, []).append({
                'car_bbox': car_bbox,
                'lp_bbox': lp_bbox,
                'car_id': car_id,
                'lp_id': lp_id
            })

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Unable to open video: {video_path}")

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Initialize the VideoWriter
    out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'avc1'), fps, (frame_width, frame_height))

    # Process the video frame by frame
    frame_number = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Draw bounding boxes for the current frame
        if frame_number in detections:
            for detection in detections[frame_number]:
                # Draw car bounding box
                car_bbox = detection['car_bbox']
                x1, y1, x2, y2 = map(int, car_bbox)
                cv2.rectangle(frame, (x1, y1), (x2, y2), CAR_COLOR, CAR_BOX_THICKNESS)
                cv2.putText(frame, f"ID {int(detection['car_id'])}", (x1, y1 - 10), FONT, FONT_SCALE, CAR_COLOR, FONT_THICKNESS)

                # Draw license plate bounding box (if available)
                lp_bbox = detection['lp_bbox']
                if lp_bbox:
                    lp_coords = lp_bbox[0]
                    lp_x1, lp_y1, lp_x2, lp_y2 = map(int, list(lp_coords))
                    cv2.rectangle(frame, (lp_x1, lp_y1), (lp_x2, lp_y2), (0, 255, 0), 1)
                    cv2.putText(frame, f"{detection['lp_id']}", (lp_x1, lp_y1 - 10), FONT, LP_FONT_SCALE, LP_COLOR, LP_FONT_THICKNESS)

        # Write the frame to the output video
        out.write(frame)

        # Increment the frame number
        frame_number += 1

    # Release resources
    cap.release()
    out.release()
    print(f"Video with bounding boxes saved to {output_video_path}")

