import re

def correction_french(detected_text: str, CHAR_TO_INT_DICT: dict, INT_TO_CHAR_DICT: dict, logs: list) -> str | None:
    """
    Takes a license plate reading and corrects it. The resulting license plate is expected to be in the format XX999XX

    Args:
        detected_text (str): A license plate reading (not yet corrected so it could be something like X9XX9XX) as some letters could be mistaken for numbers and vice versa.
        CHAR_TO_INT_DICT (dict): dictionary that maps letters to numbers. These are numbers that could be mistaken for letters like O and 0.
        INT_TO_CHAR_DICT (dict): dictionary that maps numbers to letters. These are letters that could be mistaken for numbers like 1 and I.
        logs (list): list that stores the corrected license plates. This is used for debugging purposes.

    Returns:
        str | None: the corrected license plate in french format if the detected text is valid, None otherwise.
    """    
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
    for i in range(2): # correcting first 2 letters
        if result_without_special_chars[i].isalpha():
            corrected_text += result_without_special_chars[i].upper()
        elif result_without_special_chars[i] in INT_TO_CHAR_DICT.keys():
            corrected_text += INT_TO_CHAR_DICT[result_without_special_chars[i]]
        else:
            return None
        
    
    for i in range(2,5): # correcting the 3 digits
        if result_without_special_chars[i].isdigit():
            corrected_text += result_without_special_chars[i]
        elif result_without_special_chars[i] in CHAR_TO_INT_DICT:
            corrected_text += CHAR_TO_INT_DICT[result_without_special_chars[i]]
        else:
            return None


    for i in range(5, 7): # correcting the last 2 letters
        if result_without_special_chars[i].isalpha():
            corrected_text += result_without_special_chars[i].upper()
        elif result_without_special_chars[i] in INT_TO_CHAR_DICT.keys():
            corrected_text += INT_TO_CHAR_DICT[result_without_special_chars[i]]
        else:
            return None
        
    logs.append(corrected_text)
    return corrected_text
    

def correction_british(detected_text: str, CHAR_TO_INT_DICT: dict, INT_TO_CHAR_DICT: dict) -> str | None:
    """
    Takes a license plate reading and corrects it. The resulting license plate is expected to be in the format XX99XXX

    Args:
        detected_text (str): A license plate reading (not yet corrected so it could be something like X9XX9XX) as some letters could be mistaken for numbers and vice versa.
        CHAR_TO_INT_DICT (dict): dictionary that maps letters to numbers. These are numbers that could be mistaken for letters like O and 0.
        INT_TO_CHAR_DICT (dict): dictionary that maps numbers to letters. These are letters that could be mistaken for numbers like 1 and I.

    Returns:
        str | None: the corrected license plate in british format if the detected text is valid, None otherwise.
    """    
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


def correction_license_plate_country(country: str, detected_text: str, CHAR_TO_INT_DICT: dict, INT_TO_CHAR_DICT: dict, logs: list) -> str | None:
    """ 
    Takes a license plate reading and corrects it. The resulting license plate is expected to be in the format XX99XXX or XX999XX depending on the country.

    Args:
        country (str): specifies the country of the license plate 'fr' for french or 'eng' for english.
        detected_text (str): A license plate reading (not yet corrected so it could be something like X9XX9XX) as some letters could be mistaken for numbers and vice versa.
        CHAR_TO_INT_DICT (dict): dictionary that maps letters to numbers. These are numbers that could be mistaken for letters like O and 0.
        INT_TO_CHAR_DICT (dict): dictionary that maps numbers to letters. These are letters that could be mistaken for numbers like 1 and I.
        logs (list): list that stores the corrected license plates. This is used for debugging purposes.

    Returns:
        str | None: the corrected license plate in the format XX99XXX or XX999XX (depending on the country) if the detected text is valid, None otherwise.
    """    
    if country == 'eng':
        return correction_british(detected_text, CHAR_TO_INT_DICT, INT_TO_CHAR_DICT)
    elif country == 'fr':
        return correction_french(detected_text, CHAR_TO_INT_DICT, INT_TO_CHAR_DICT, logs)