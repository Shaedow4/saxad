import numpy as np
from tensorflow import keras
from typing import Union

def data_normalization_min_max(to_be_normalized_data: np.array) -> np.array:
    max = np.max(to_be_normalized_data, axis=0)
    min = np.min(to_be_normalized_data, axis=0)
    return (to_be_normalized_data - min)/(max-min+0.00001)  


def dynamic_modify_tensor_shape(myInput, window_size):
    op = keras.backend.tile(myInput, (window_size-1, 1))
    new_shape = keras.backend.concatenate(
        [(-1, window_size-1), keras.backend.shape(myInput)[1:2]], axis=0)
    return keras.backend.reshape(op, new_shape)


def windowed_dataset(x: np.array, window_size: int, feature_amount: int) -> Union[np.array, np.array]:
    training_dataset = np.empty(shape=(len(x)-window_size, window_size-1, feature_amount), dtype=np.float64)
    training_correct_results = np.empty(shape=(len(x)-window_size, feature_amount), dtype=np.float64)
    counter = 0
    for i in range(window_size, len(x)):
        training_dataset[counter] = x[i-window_size: i-1]
        training_correct_results[counter] = x[i].reshape((feature_amount)) 
        counter = counter + 1
    return training_dataset, training_correct_results
