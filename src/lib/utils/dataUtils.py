import numpy as np
from PIL import Image as im
from typing import Union, Tuple

class DataUtils:
    '''
    Utility methods for importing and exporting data, as well
    as for simple pre-processing.
    '''
    def __init__(self, input_dir_path: str, output_dir_path: str):
        self.input_dir_path = input_dir_path
        self.output_dir_path = output_dir_path

    def load_npy(self, filename: str) -> np.array:
        return np.load(
            self.input_dir_path + f'/{filename}.npy',
            allow_pickle=True
        )
    
    def load_data(self, train: bool) -> Union[Tuple[np.array, np.array], np.array]:
        if train:
            return (self.load_npy('trainX'), self.load_npy('trainy'))
        else:
            return self.load_npy('testX')

    def save_model(self, filename: str):
        raise NotImplementedError   

    def display_image(self, array: np.array) -> im.Image: 
        return im.fromarray(array)
    
    def _normalize(self, array: np.array) -> np.array:
        return (array / 255)
    
    def _standardize(self, array: np.array) -> np.array:
        mean, std = np.mean(
            array, 
            axis=(1,2), 
            keepdims=True
        ), np.std(
            array,
            axis=(1,2), 
            keepdims=True
        )
        return ((array - mean) / std)