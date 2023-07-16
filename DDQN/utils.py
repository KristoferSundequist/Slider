from typing import *
import numpy as np

def get_two_hot(value: float, min_value: float, max_value: float, num_buckets: int) -> List[float]:
    if value < min_value:
        raise Exception("value unexpectedly less than min_value")
    
    if value > max_value:
        raise Exception("value unexpectedly greater than max_value")
    
    values = list(np.linspace(min_value, max_value, num_buckets))
    print(values)

    min_index = 0
    max_index = 0
    min_value = 0
    max_value = 0
    for i in range(num_buckets):
        if values[i+1] > value:
            min_index = i
            max_index = i+1
            min_value = values[min_index]
            max_value = values[max_index]
            break

    min_value_weight = abs(max_value - value) / (max_value - min_value)
    max_value_weight = abs(min_value - value) / (max_value - min_value)
    
    two_hot = [0 for i in range(num_buckets)]
    two_hot[min_index] = min_value_weight
    two_hot[max_index] = max_value_weight
    return two_hot
    