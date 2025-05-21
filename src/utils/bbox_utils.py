

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