from collections import Counter
import yaml

# # The dictionary
# l_dict = { 
#   'AK64MMV': 1,
#   'AK64DMV': 79,
#   'AK64OMV': 3,
#   'AN64DMV': 2,
#   'AK64DAV': 1,
#   'AK54DMV': 1,
#   'AK64DAC': 1,
#   'AK64DHV': 1,
#   'AR64DMV': 1
# }

def extract_license_plate(lp_dict):
    """
    For a given dictionary of possible license plates (with their occurences) for a given car, 
    find the most likely license plate, Example for car with the id 2
    2:
        JA13NRU: 1
        NA13MRU: 3
        NA13NRU: 6
    Most likely license plate: NA13NRU
    The algorithm works by finding the most occurring letter/number in each position of the strings.
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


def correct_all_license_plates(car_and_lp):
    """
    Returns a dictionary with the cars ids (keys) and their corresponding license plates (values)
    """
    corrected_license_plates = {}
    for car_id, license_plate_ids_dict in car_and_lp.items():
        result_lp = extract_license_plate(license_plate_ids_dict)
        corrected_license_plates[car_id] = result_lp
    return corrected_license_plates


if __name__ == '__main__':
    with open("output_test1.yaml", "r") as file:
        data = yaml.safe_load(file)
    print(data)
    print("\n ############################## \n")
    license_plate_ids_dict = correct_all_license_plates(data)
    print(license_plate_ids_dict)
