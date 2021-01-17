import cv2
import pandas as pd
import numpy as np
# from keras.preprocessing.image import ImageDataGenerator


def resize_image(image):
    height = 220
    width = 220
    dimensions = (width, height)
    resized_image = cv2.resize(
        image, dimensions, interpolation=cv2.INTER_CUBIC)

    return resized_image


def reduce_image_noise(image):
    return cv2.GaussianBlur(image, (5, 5), 0)


def normalise_image(image):
    return image/255


# def augment_data(training_data):
#     ImageDataGenerator(
#         featurewise_center=False,  # set input mean to 0 over the dataset
#         samplewise_center=False,  # set each sample mean to 0
#         featurewise_std_normalization=False,  # divide inputs by std
#         samplewise_std_normalization=False,  # divide each input by its std
#         zca_whitening=False,  # apply ZCA whitening
#         rotation_range=30,  # randomly rotate images in the range
#         zoom_range=0.2, # Randomly zoom image
#         width_shift_range=0.1,  # randomly shift images horizontally
#         height_shift_range=0.1,  # randomly shift images vertically
#         horizontal_flip=True,  # randomly flip images
#         vertical_flip=False  # randomly flip images
#     )


def read_and_process_image(path):
    image = cv2.imread(path)
    image = resize_image(image)
    image = reduce_image_noise(image)
    image = normalise_image(image)

    return image


def create_data_set():
    df = pd.read_csv('data/final_image_data.csv')
    X = []
    y = []

    for idx, row in df.iterrows():
        try:
            image = read_and_process_image(row['image_url'])
            X.append(image)
            y.append(1 if row['target'] == 'with_mask' else 0)
        except Exception:
            print(row['image_url'])

    X = np.asarray(X)
    y = np.asarray(y)

    np.save('data/processed_data_X.npy', X)
    np.save('data/processed_data_y.npy', y)


if __name__ == '__main__':
    create_data_set()
