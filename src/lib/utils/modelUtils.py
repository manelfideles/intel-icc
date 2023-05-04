import wandb
import tensorflow as tf
from wandb.keras import WandbCallback

class ModelUtils:
    '''
    Utility methods for model training, 
    validation, testing and selection.
    '''
    def __init__(self, model, training_data):
        self.model = model
        self.training_data = training_data
        # self.device = torch_device('cuda' if cuda.is_available() else 'cpu')

    def build_optimizer(self, optimizer: str = 'adam', lr: float = 0.01, momentum: float = 0.0):
        if optimizer == 'adam':
            return tf.keras.optimizers.Adam(learning_rate=lr)
        else:
            return tf.keras.optimizers.SGD(learning_rate=lr, momentum=momentum)
    
    def build_model(self, config):
        '''
        Builds necessary models components according to 
        config and compiles them into self.model.
        :param config: config passed from config file
        '''
        optimizer = self.build_optimizer(config.optimizer, config.learning_rate, 0.9)
        loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
        metrics = ['Accuracy', 'AUC', 'Precision', 'Recall']
        self.model.compile(optimizer, loss, metrics=metrics)
        print(self.model.summary())

    def train(self):
        '''
        This method trains self.model. 
        Performs validation in an embedded manner.
        @TODO Add cross-validation ?
        :param x: training data
        :param y: training labels
        '''
        x, y = self.training_data
        config_defaults = {
            'optimizer': 'adam',
            'batch_size': 16,
            'learning_rate': 0.01,
            'epochs': 10,
        }
        wandb.init(project='intel-icc', entity='manelfideles', config=config_defaults)
        config = wandb.config
        self.build_model(config)
        self.model.fit(
            x, y,
            validation_split=0.2,
            epochs=config.epochs, 
            batch_size=config.batch_size,
            callbacks=[WandbCallback()]
        )
        wandb.finish()