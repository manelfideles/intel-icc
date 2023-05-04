from typing import Dict, Any

sweep_config: Dict[str, Any] = {
    'method': 'random',
    'metric': {
        'goal': 'minimize', 
        'name': 'loss'
    },
    'parameters': {
        'optimizer': {
            'values': ['adam', 'sgd']
        },
        'epochs': { 'value': 10 },
        'batch_size': {
            'values': [32, 64, 128]
        },
        'learning_rate': {
            'values': [0.1, 0.01, 0.001, 0.0001]
        }
    }
}