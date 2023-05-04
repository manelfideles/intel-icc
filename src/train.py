# Imports and Setup
import wandb
from lib.utils.modelUtils import ModelUtils
from lib.utils.dataUtils import DataUtils
from lib.utils.sweep_configs import sweep_config
from lib.models.CNN import cnns
import random
import numpy as np
import tensorflow as tf
import keras.backend as K
from tensorflow.keras.utils import to_categorical
from os import environ

# Set random seeds
environ['TF_CUDNN_DETERMINISTIC'] = '1'
random.seed(hash('setting random seeds') % 2**32 - 1)
np.random.seed(hash('improves reproducibility') % 2**32 - 1)
tf.random.set_seed(hash('so that runs are repeatable'))

print("# GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
if len(tf.config.list_physical_devices('GPU')):
    print('GPU Device Name', tf.test.gpu_device_name())

LABELS = [
    'buildings', 
    'forest', 
    'glacier', 
    'mountain', 
    'sea', 
    'street'
]

# Import
data_utils = DataUtils('../data', '../outputs')
x_train, y_train = data_utils.load_data(train=True)
x_test = data_utils.load_data(train=False)

# Pre-process
x_train = data_utils._standardize(
    data_utils._normalize(x_train)
)
y_train = to_categorical(
    y_train, len(LABELS)
)

K.clear_session()
model = ModelUtils(
    model=cnns['le-net-5'],
    training_data=(x_train, y_train)
)

wandb.login()
sweep_id = wandb.sweep(sweep_config, entity='manelfideles', project="intel-icc")
wandb.agent(
    sweep_id, 
    model.train,
    count = 5
)