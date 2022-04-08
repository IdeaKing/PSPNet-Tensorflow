import tensorflow as tf
import matplotlib.pyplot as plt

import dataset

if __name__ == "__main__":
    DATASET_PATH = "data/datasets/cropped_refuge"

    file_names = dataset.load_data(dataset_path=DATASET_PATH)
    ds = dataset.Dataset(file_names, dataset_path=DATASET_PATH)()

    for image, mask in ds:
        fig = plt.figure(figsize=(10, 10))
        fig.add_subplot(1, 2, 1)
        plt.imshow(image[0])
        plt.title("Optic Disk Region")
        fig.add_subplot(1, 2, 2)
        plt.imshow(mask[0])
        plt.title("Semantic Map")
        plt.show()
        break
