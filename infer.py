import argparse
import os

import numpy as np
import cv2
from tensorflow.keras.models import load_model
from face_recognition import face_locations

from preprocessing import process_image


def infer(model_path, image):
    face_coordinates = face_locations(image, model='cnn')

    if len(face_coordinates) == 0:
        print('Could not find any faces in the image')
        return

    model = load_model(model_path)

    processed_images = np.asarray(
        [
            process_image(image[y_min:y_max, x_min:x_max])
            for y_min, x_max, y_max, x_min in face_coordinates
        ]
    )
    predictions = model.predict(processed_images)

    return zip(face_coordinates, np.concatenate(predictions).ravel())


def display_predictions(image, predictions):
    returned_image = np.array(image, copy=True)

    for face, prediction in predictions:
        y_min, x_max, y_max, x_min = face
        classification = np.rint(prediction)

        colour = (0, 255, 0) if classification == 1 else (0, 0, 255)
        cv2.rectangle(
            returned_image, (x_min, y_min), (x_max, y_max), colour, 2
        )
        cv2.putText(
            returned_image,
            f'{prediction:.5f}',
            (x_min + 5, y_max - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            colour,
            1,
        )

    cv2.imshow('Prediction', returned_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=(
            'Make an inference on an image if the faces in the image are '
            'wearing face masks',
        )
    )
    parser.add_argument(
        'model',
        help='The path for the model to make the inference',
    )
    parser.add_argument(
        'image',
        help='The image to make the inference on'
    )
    args = parser.parse_args()

    try:
        assert os.path.exists(args.model)
    except AssertionError:
        print('The path to the model does not exist')
        exit(1)

    try:
        assert os.path.exists(args.image)
    except AssertionError:
        print('The path to the image does not exist')
        exit(1)

    image = cv2.imread(args.image)
    predictions = infer(args.model, image)
    display_predictions(image, predictions)
