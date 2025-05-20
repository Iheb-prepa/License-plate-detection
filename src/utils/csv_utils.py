import yaml
import csv
import postprocess_results 
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


def generate_corrected_csv(YAML_FILE_PATH: "Path", csv_output_path: "Path") -> None:
    """ 
    Correct the CSV file with the most correct license plate number for each vehicle by 
    reading the YAML file, extracting the most likely license plate for each car, and update the CSV file accordingly.

    Args:
        YAML_FILE_PATH (Path): _description_
        csv_output_path (Path): _description_
    """    
    with open(YAML_FILE_PATH, "r") as file:
        data = yaml.safe_load(file)

    # Extract the most occurring (most likely) license plate value for each detected license plate
    license_plate_ids_dict = postprocess_results.correct_all_license_plates(data)
    # print(license_plate_ids_dict)

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


def update_lp_data(csv_data_dict: dict, frame_number: int, car_id: list, lp_id: int, lp_bbox: list, lp_confidence: float) -> None:
    """
    Updates the dictionary csv_data_dict with the detected objects information. The csv_data_dict will be later saved into a csv file.

    Args:
        csv_data_dict (dict): the dictionary csv_data_dict that has (frame_number, int(track_id)) as key and [frame_number, int(car_id), str(car_bbox), str(lp_id), str(lp_bbox), lp_confidence]
        frame_number (int): Number of the frame
        car_id (list): Id of the detected car
        lp_id (int): Id of the detected license plate
        lp_bbox (list): Bounding box of the detected license plate
        lp_confidence (float): Confidence score of the detected license plate
    """    
    if (frame_number, car_id) in csv_data_dict:
        row = csv_data_dict[(frame_number, car_id)] # car_bbox should already be in csv_data_dict
        row[3] = str(lp_id)
        row[4] = str(lp_bbox)  # Update lp_bbox
        row[5] = lp_confidence  # Update lp_confidence
        csv_data_dict[(frame_number, car_id)] = row
