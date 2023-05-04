from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Flatten, Dense, Dropout

cnns = {
    'simple-net': Sequential([
        Conv2D(32, 3, padding='same', input_shape=(50, 50, 3), activation='relu'),
        MaxPooling2D(),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(units=6, activation='softmax')
    ]),
    'conv-conv-mp-flat-ds-ds-do-ds': Sequential([
        Conv2D(16, 3, padding='same', input_shape=(50, 50, 3), activation='relu'),
        Conv2D(16, 5, padding='same', activation='relu'),
        MaxPooling2D(2),
        Flatten(),
        Dense(100, activation='relu'),
        Dense(50, activation='relu'),
        Dropout(rate=0.6),
        Dense(units=6, activation='softmax')
    ]),
    'conv-conv-mp-f-ds-do-ds-do-ds': Sequential([
        Conv2D(16, 3, strides=1, padding="same", input_shape=(50, 50, 3), activation='relu'),
        Conv2D(32, 3, strides=1, padding="same", activation='relu'),
        MaxPooling2D(2),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(rate=0.6),
        Dense(64, activation='relu'),
        Dropout(rate=0.5),
        Dense(units=6, activation='softmax')
    ]),
    'le-net-5': Sequential([
        Conv2D(filters=6, kernel_size=(5, 5), activation='relu', input_shape=(50, 50, 3)),
        AveragePooling2D(pool_size=(2, 2)),
        Conv2D(filters=16, kernel_size=(5, 5), activation='relu'),
        AveragePooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(units=120, activation='relu'),
        Dense(units=84, activation='relu'),
        Dense(units=6, activation='softmax')
    ])
}