import os

from bs4 import BeautifulSoup
import pandas as pd
import cv2


DATA_DIR = 'data'
ANNOTATION_DIR = os.path.join(DATA_DIR, 'annotations')
IMAGE_DIR = os.path.join(DATA_DIR, 'images')
NEW_IMAGES_DIR = os.path.join(DATA_DIR, 'face_images')


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
                counter += 1

    # Save df to csv
    df.to_csv(os.path.join(DATA_DIR, 'image_data.csv'), index=False)


if __name__ == '__main__':
    main()
