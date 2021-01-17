import os
import logging

from bs4 import BeautifulSoup
import pandas as pd
import cv2


DATA_DIR = 'data'
ANNOTATION_DIR = os.path.join(DATA_DIR, 'annotations')
IMAGE_DIR = os.path.join(DATA_DIR, 'images')
NEW_IMAGES_DIR = os.path.join(DATA_DIR, 'face_images')
NO_MASK_IMAGE_DIR = os.path.join(DATA_DIR, 'no_mask_face_images')


logger = logging.getLogger("Preparing images")
logger.setLevel(logging.INFO)

ch = logging.StreamHandler()
ch.setLevel(logging.INFO)

formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

ch.setFormatter(formatter)
logger.addHandler(ch)


def isolate_face_from_coordinates(image, x_min, x_max, y_min, y_max):
    """Isolate the face in the image from the coordinates"""
    return image[y_min:y_max, x_min:x_max]


def get_coordinates_from_object(obj):
    """Get the coordinates of the face from the beautiful soup object."""
    x_min = int(obj.find('xmin').text)
    x_max = int(obj.find('xmax').text)
    y_min = int(obj.find('ymin').text)
    y_max = int(obj.find('ymax').text)

    return x_min, x_max, y_min, y_max


def get_target_from_object(obj):
    """Return the target variable from the beautiful soup object."""
    return obj.find('name').text


def main():
    df = pd.DataFrame(columns=['id', 'target', 'image_url'])
    counter = 0
    for annotation_file in os.listdir(ANNOTATION_DIR):
        with open(os.path.join(ANNOTATION_DIR, annotation_file), 'r') as f:
            content = f.read()
            soup = BeautifulSoup(content, 'lxml')

            # Data about each face is in the object tag
            objects = soup.find_all('object')

            # The coresponding image has the same file name in the images dir
            image_url = os.path.join(
                IMAGE_DIR,
                '.'.join([os.path.splitext(annotation_file)[0], 'png'])
            )
            image = cv2.imread(image_url)

            for obj in objects:
                coordinates = get_coordinates_from_object(obj)
                target = get_target_from_object(obj)

                face_image = isolate_face_from_coordinates(
                    image, *coordinates
                )

                if not os.path.exists(NEW_IMAGES_DIR):
                    os.makedirs(NEW_IMAGES_DIR)

                new_image_url = os.path.join(NEW_IMAGES_DIR, f'{counter}.png')
                cv2.imwrite(new_image_url, face_image)

                df.loc[counter] = [counter, target, new_image_url]

                if counter % 250 == 0:
                    logger.info(f'Added {counter} images to dataframe')

                counter += 1

    # Save df to csv
    df.to_csv(os.path.join(DATA_DIR, 'image_data.csv'), index=False)

    # Add images from second data set
    for image_file in os.listdir(NO_MASK_IMAGE_DIR):
        df.loc[counter] = [
            counter,
            'without_mask',
            os.path.join(NO_MASK_IMAGE_DIR, image_file)
        ]

        if counter % 250 == 0:
            logger.info(f'Added {counter} images to dataframe')

        counter += 1

    # Save df to csv
    df.to_csv(os.path.join(DATA_DIR, 'all_image_data.csv'), index=False)

    # Remove mask worn incorrect type
    df = df[df['target'] != 'mask_weared_incorrect']
    df.to_csv(os.path.join(DATA_DIR, 'final_image_data.csv'), index=False)


if __name__ == '__main__':
    main()
