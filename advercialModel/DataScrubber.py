import re
from scipy.spatial import distance
def pad_to_equal_length(s1, s2):
    """
    Pad the shorter of two strings with spaces to make them equal in length.
    """
    max_length = max(len(s1), len(s2))
    return s1.ljust(max_length), s2.ljust(max_length)

def similarity_remove(strings, min_hamming_distance):
    """
    Remove strings from a list until all remaining strings
    have a Hamming distance of at least min_hamming_distance.

    Args:
        strings (list of str): The input list of strings.
        min_hamming_distance (int): The minimum Hamming distance required between strings.

    Returns:
        list of str: A filtered list of strings.
    """

    filtered_strings = []

    for s in strings:
        # Check if the string has a sufficient Hamming distance from all others in the filtered list
        if all(distance.hamming(list(pad_to_equal_length(s, other)[0]),
                                list(pad_to_equal_length(s, other)[1])) * max(len(s), len(other)) >= min_hamming_distance
               for other in filtered_strings):
            filtered_strings.append(s)

    return filtered_strings




def similarity_remove_string_dict(data:dict[str:int], min_hamming_distance_ratio=0.4):
    """
    Remove strings from a dictionary until all remaining strings
    have a Hamming distance of at least min_hamming_distance.

    Args:
        data (dict): A dictionary where keys are strings, and values are integers or floats.
        min_hamming_distance (int): The minimum Hamming distance required between strings.

    Returns:
        dict: A filtered dictionary containing strings that meet the Hamming distance requirement.
    """

    filtered_data = {}

    for current_string, current_value in data.items():
        # Check if the current string meets the Hamming distance constraint for all others in filtered_data
        if all(
            distance.hamming(
                list(pad_to_equal_length(current_string, other_string)[0]),
                list(pad_to_equal_length(current_string, other_string)[1])
            ) >= min_hamming_distance_ratio
            for other_string in filtered_data
        ):
            filtered_data[current_string] = current_value
        else:
            # If not, check if the current string's value is higher than an existing one with the same key
            for other_string in list(filtered_data):
                padded_current, padded_other = pad_to_equal_length(current_string, other_string)
                if (
                    distance.hamming(list(padded_current), list(padded_other)) < min_hamming_distance_ratio
                    and current_value > filtered_data[other_string]
                ):
                    del filtered_data[other_string]
                    filtered_data[current_string] = current_value

    return filtered_data

def remove_non_chinese_answers(strings,min_ratio_to_remove=0.5):
    """
    Removes strings that have a ratio of a-zA-Z characters to the total number of characters greater than 0.5.

    Args:
        strings (list of str): A list of strings.

    Returns:
        list of str: A filtered list of strings that are likely Chinese or have fewer Latin characters.
    """
    filtered_strings = []
    for s in strings:
        total_chars = len(s)
        latin_chars = len(re.findall(r'[a-zA-Z]', s))
        if latin_chars / total_chars <= min_ratio_to_remove:
            filtered_strings.append(s)
    return filtered_strings

def scrub_chinese_string_answers(string_list:list[str],min_hamming_distance=6)->list[str]:
    chinese_answers =remove_non_chinese_answers(string_list)
    result=similarity_remove(chinese_answers,min_hamming_distance=min_hamming_distance)
    return result

if __name__ == '__main__':
    strings = [

        "This is an English sentence.",

        "这是一个中文句子。",

        "Mixed 语言 sentence 中文。",
        "Mixed 语言 sentence 中文。",
        "Mixed 语言 sentence 中文a。",
        "Mixed 语言。",

        "Another purely English example."

    ]

    data = {
        "": 0.1,
        "现在广大男性同胞清醒了，反击了，然后她们就开始在这讲道理了。当时女权运动大肆讨论男性问题的时候，声称男性需要改变的时候，她们人呢？为什么没见她们出来跟女权运动解释现实中男人和女人都需要相互理解和1": 9.0,
        "现在广大男性同胞清醒了，反击了，然后她们就开始在这讲道理了。当时女权运动大肆讨论男性问题的时候，声称男性需要改变的时候，她们人呢？为什么没见她们出来跟女权运动解释现实中男人和女人都需要adqwqqqq": 8.0,
        "现在广大男性同胞清醒了，反击了，然后她们就开始在这讲道理了。当时女权运动大肆讨论男性问题的时候，声称男性需要改变的时候，她们人呢？为什么没见她们出来跟女权运动解释现实中男人和女人都需要相互理解和尊": 10.0,
        "现在广大男性同胞清醒了，反击了，然后她们就开始在这讲道理了。当时女权运动大肆讨论男性问题的时候，声称男性需要改变的时候，她们人呢？为什么没见她们出来跟女权运动解释现实aaaaa": 7.0,
    }
    min_hamming_distance = 0.10

    result = similarity_remove_string_dict(data, min_hamming_distance)
    print(result)
    print("length:",result.__len__())

    filtered_strings = scrub_chinese_string_answers(strings,10)
    print(filtered_strings)