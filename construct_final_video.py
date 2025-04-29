import cv2
import ast  # To parse the bbox strings as lists
import csv


def interpolate_bboxes(prev_bbox, next_bbox, frame_number, prev_frame, next_frame):
    """Interpolate the bounding box between two frames."""
    if prev_bbox is None or next_bbox is None:
        return None
    x1_prev, y1_prev, x2_prev, y2_prev = prev_bbox[0]
    x1_next, y1_next, x2_next, y2_next = next_bbox[0]
    # print(next_frame, frame_number, prev_frame)
    if next_frame == prev_frame:
        return None
    # Linear interpolation of coordinates
    x1 = int(x1_prev + (x1_next - x1_prev) * (frame_number - prev_frame) / (next_frame - prev_frame))
    y1 = int(y1_prev + (y1_next - y1_prev) * (frame_number - prev_frame) / (next_frame - prev_frame))
    x2 = int(x2_prev + (x2_next - x2_prev) * (frame_number - prev_frame) / (next_frame - prev_frame))
    y2 = int(y2_prev + (y2_next - y2_prev) * (frame_number - prev_frame) / (next_frame - prev_frame))
    
    return [x1, y1, x2, y2]


def zoom_on_cropped_lp(frame, lp_x1, lp_x2, lp_y1, lp_y2):
    scale_factor = 1.5
    cropped_region = frame[lp_y1:lp_y2, lp_x1:lp_x2]
    zoomed_region = cv2.resize(
        cropped_region, 
        None, 
        fx=scale_factor, 
        fy=scale_factor, 
        interpolation=cv2.INTER_LINEAR
    )

    zoom_h, zoom_w = zoomed_region.shape[:2]

    center_x = (lp_x1 + lp_x2) // 2
    center_y = (lp_y1 + lp_y2) // 2

    new_x1 = max(0, center_x - zoom_w // 2)
    new_y1 = max(0, center_y - zoom_h // 2)
    new_x2 = min(frame.shape[1], new_x1 + zoom_w)
    new_y2 = min(frame.shape[0], new_y1 + zoom_h)

    # Adjust zoomed_region if it goes out of bounds
    zoomed_region = zoomed_region[:new_y2 - new_y1, :new_x2 - new_x1]
    
    return zoomed_region, new_x1, new_y1, new_x2, new_y2


def main(csv_path, video_path, output_video_path):

    # Parameters for drawing
    car_color = (255, 0, 0)  # Blue for car bounding boxes
    lp_color = (0, 255, 0)   # Green for license plate bounding boxes
    CAR_BOX_THICKNESS = 5
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.5
    font_thickness = 5

    lp_font_scale = 1.5
    lp_font_thickness = 4

    # Load the CSV data into memory
    detections = {}
    with open(csv_path, mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        for row in csv_reader:
            # print(row)
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
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec

    out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'avc1'), fps, (frame_width, frame_height))

    # List frames where bounding boxes are available (for interpolation)
    all_frame_numbers = sorted(detections.keys())

    # Process the video frame by frame
    frame_number = 0
    prev_frame = None
    next_frame = None
    prev_lp_bbox = None
    next_lp_bbox = None

    # Store frames with license plates for interpolation
    available_lp_frames = {frame: [] for frame in all_frame_numbers}
    for frame in all_frame_numbers:
        for detection in detections[frame]:
            if detection['lp_bbox'] is not None:
                available_lp_frames[frame].append(detection['lp_bbox'])

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Find the closest previous and next frames with license plate bounding boxes
        prev_lp_bbox = None
        next_lp_bbox = None
        prev_frame = None
        next_frame = None

        # Loop through all frames to find the closest frames with license plate bounding boxes
        for i in range(len(all_frame_numbers)):
            if all_frame_numbers[i] <= frame_number:
                prev_frame = all_frame_numbers[i]
                prev_lp_bbox = available_lp_frames[prev_frame][0] if available_lp_frames[prev_frame] else None
            if all_frame_numbers[i] > frame_number:
                next_frame = all_frame_numbers[i]
                next_lp_bbox = available_lp_frames[next_frame][0] if available_lp_frames[next_frame] else None
                break

        # Draw bounding boxes for the current frame
        if frame_number in detections:
            for detection in detections[frame_number]:
                # Draw car bounding box
                car_bbox = detection['car_bbox']
                # print("GOT CAR BBOX", car_bbox)
                x1, y1, x2, y2 = map(int, car_bbox)
                cv2.rectangle(frame, (x1, y1), (x2, y2), car_color, CAR_BOX_THICKNESS)
                cv2.putText(frame, f"ID {int(detection['car_id'])}", (x1, y1 - 10), font, font_scale, car_color, font_thickness)

                # Draw license plate bounding box (if available)
                lp_bbox = detection['lp_bbox']
                if lp_bbox:
                    # print("GOT BBOX", lp_bbox)
                    lp_x1, lp_y1, lp_x2, lp_y2 = map(int, list(lp_bbox[0]))
                    zoomed_region, new_x1, new_y1, new_x2, new_y2 = zoom_on_cropped_lp(frame, lp_x1, lp_x2, lp_y1, lp_y2)
                    frame[new_y1:new_y2, new_x1:new_x2] = zoomed_region
                    cv2.rectangle(frame, (new_x1, new_y1), (new_x2-1, new_y2-1), lp_color, 3)
                    # cv2.putText(frame, f"{detection['lp_id']}", (lp_x1, lp_y1 - 10), font, lp_font_scale, lp_color, lp_font_thickness)
                    cv2.putText(frame, f"{detection['lp_id']}", (new_x1, new_y1 - 10), font, lp_font_scale, lp_color, lp_font_thickness)
                    # prev_lp_bbox = lp_bbox
                    # prev_frame = frame_number
                # else:
                #     # If there is no license plate bbox, try to interpolate
                #     if prev_lp_bbox is not None and next_lp_bbox is not None:
                #         interpolated_bbox = interpolate_bboxes(prev_lp_bbox, next_lp_bbox, frame_number, prev_frame, next_frame)
                #         if interpolated_bbox:
                #             x1, y1, x2, y2 = map(int, list(interpolated_bbox))
                #             cv2.rectangle(frame, (x1, y1), (x2, y2), lp_color, box_thickness)
                #             cv2.putText(frame, f"LP {detection['lp_id']}", (x1, y1 - 10), font, lp_font_scale, lp_color, lp_font_thickness)

        # Write the frame to the output video
        out.write(frame)

        # Increment the frame number
        frame_number += 1

        # # Check the next frame for interpolation
        # if frame_number in detections:
        #     next_frame = frame_number
        #     for detection in detections[frame_number]:
        #         next_lp_bbox = detection['lp_bbox']

    # Release resources
    cap.release()
    out.release()
    print(f"Video with bounding boxes saved to {output_video_path}")


if __name__ == '__main__':
    # Paths to video and output video file
    video_path = 'test_videos/test1.mp4'
    output_video_path = 'output_with_bboxes_1.mp4'
    # csv_path = 'detections_test1.csv'
    csv_path = 'interpolated_lp_detections_test1.csv'
    csv_path = 'after_interpolation_test1.csv'
    main(csv_path, video_path, output_video_path)