from collections import Counter
import yaml

def extract_license_plate(lp_dict: dict) -> str:
    """
    For a given dictionary of possible license plates (with their occurences) for a given car, 
    find the most likely license plate, Example for car with the id 2
    2:
        JA13NRU: 1
        NA13MRU: 3
        NA13NRU: 6
    Most likely license plate: NA13NRU
    The algorithm works by finding the most occurring letter/number in each position of the strings.

    Args:
        lp_dict (dict): dictionary with the read license plates (keys) and their occurences (values).

    Returns:
        str: the most likely license plate.
    """    
    # Find the maximum key length
    max_length = max(len(key) for key in lp_dict.keys())

    # Create a list to store the character counters for each position
    position_counters = [Counter() for _ in range(max_length)]

    # Count character occurrences for each position
    for key, count in lp_dict.items():
        for i, char in enumerate(key):
            position_counters[i][char] += count

    # Construct the resulting string
    result_lp = ''.join(counter.most_common(1)[0][0] for counter in position_counters)
    return result_lp


def correct_all_license_plates(car_and_lp: dict) -> dict:
    """ 
    Returns a dictionary with the cars ids (keys) and their corresponding license plates (values)

    Args:
        car_and_lp (dict): dictionary that stores license plate readings and their occurrences and their associated cars.

    Returns:
        dict: dictionary with the cars ids (keys) and their corresponding license plates (values)
    """    
    corrected_license_plates = {}
    for car_id, license_plate_ids_dict in car_and_lp.items():
        result_lp = extract_license_plate(license_plate_ids_dict)
        corrected_license_plates[car_id] = result_lp
    return corrected_license_plates
