import cv2
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
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


def create_data_sets():
    df = pd.read_csv('data/final_image_data.csv')
    X = []
    y = []

    print(f'Processing {len(df)} images')

    counter = 1
    for idx, row in df.iterrows():
        try:
            image = read_and_process_image(row['image_url'])
            X.append(image)
            y.append(1 if row['target'] == 'with_mask' else 0)
        except Exception:
            print(f"Exception raised for {row['image_url']}")

        if counter % 500 == 0:
            print(f'Processed {counter} images')

        counter += 1

    X = np.asarray(X)
    y = np.asarray(y)

    print('Finished processing images\nCreating train and test sets')

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    np.save('data/X_train.npy', X_train)
    np.save('data/X_test.npy', X_test)
    np.save('data/y_train.npy', y_train)
    np.save('data/y_test.npy', y_test)


if __name__ == '__main__':
    create_data_sets()
