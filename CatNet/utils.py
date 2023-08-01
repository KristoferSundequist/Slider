from typing import *
import math

def get_two_hot(value: float, total_min_value: float, total_max_value: float) -> List[float]:
    num_buckets = (total_max_value - total_min_value)+1

    if value < total_min_value:
        raise Exception("value unexpectedly less than min_value")

    if value > total_max_value:
        raise Exception("value unexpectedly greater than max_value")

    min_value = math.floor(value)
    max_value = min_value + 1
    min_index = min_value - total_min_value
    max_index = min_index + 1

    min_value_weight = abs(max_value - value)
    max_value_weight = abs(min_value - value)

    two_hot = [0 for i in range(num_buckets)]
    two_hot[min_index] = min_value_weight
    two_hot[max_index] = max_value_weight
    return two_hot
    