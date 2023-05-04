import numpy as np
from PIL import Image as im
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

class DataUtils:
    '''
    Utility methods for importing and exporting data, as well
    as for simple pre-processing.
    '''
    def __init__(self, input_dir_path: str, output_dir_path: str, batch_size: int = 512):
        self.input_dir_path = input_dir_path
        self.output_dir_path = output_dir_path
        self.batch_size = batch_size

    def load_npy(self, filename: str) -> np.array:
        return np.load(
            self.input_dir_path + f'/{filename}.npy',
            allow_pickle=True
        )

    def tvt_split(self, x, y, test_size: float = 0.3):
        x_train, x_test, y_train, y_test = train_test_split(
            x, y,
            test_size=test_size, 
            random_state=1
        )
        return x_train, x_test, y_train, y_test

    def make_image_generator(self, train: bool = True):
        return ImageDataGenerator(
            rescale = 1./255,
            rotation_range = 20,
            width_shift_range = 0.2,
            height_shift_range = 0.2,
            horizontal_flip = True,
            validation_split = 0.2
        )
    
    def load_data(self, train: bool):
        if train:
            x, y = self.load_npy('trainX'), to_categorical(self.load_npy('trainy'))
            x_train, x_val, y_train, y_val = self.tvt_split(x, y)
            train_datagen = self.make_image_generator()
            train_datagen.fit(x_train)
            std_datagen = ImageDataGenerator(
                preprocessing_function=train_datagen.standardize
            )
            train_data = train_datagen.flow(
                x_train,
                y_train,
                batch_size=self.batch_size,
            )
            val_data = std_datagen.flow(x_val, y_val, batch_size=self.batch_size)
            return (train_data, val_data)
        else:
            return None

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