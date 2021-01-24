from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense,
    Dropout,
)

from preprocessing import augment_data
from utils import load_data_sets
from config import Config


def train_model(training_data, test_data, steps_per_epoch, epochs=10):
    # Build model
    model = Sequential([
        Conv2D(
            32, (3, 3),
            activation='relu',
            input_shape=(Config.IMAGE_WIDTH, Config.IMAGE_HEIGHT, 3)
        ),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        Flatten(),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])

    # Compile model
    model.compile(
        'adam',
        loss='binary_crossentropy',
        metrics=['accuracy'],
    )

    # Fit model
    model.fit(
        training_data,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        validation_data=test_data
    )

    return model


def main():
    X_train, y_train, X_test, y_test = load_data_sets(
        'data/X_train.npy',
        'data/y_train.npy',
        'data/X_test.npy',
        'data/y_test.npy'
    )
    datagen = augment_data(X_train)
    batch_size = 50
    model = train_model(
        datagen.flow(X_train, y_train, batch_size=batch_size),
        (X_test, y_test),
        len(X_train)/batch_size
    )
    model.save('models/keras_cnn')


if __name__ == '__main__':
    main()
