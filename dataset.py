import os
import random
import shutil as sh

import numpy as np
import tensorflow as tf
import albumentations as A

from typing import List, Tuple


class Dataset(object):
    def __init__(self,
                 file_names: List,
                 dataset_path: str,
                 batch_size: int = 1,
                 shuffle_size: int = 64,
                 images_dir: str = "images",
                 labels_dir: str = "labels",
                 image_dims: Tuple = (512, 512),
                 augment_ds: bool = False,
                 dataset_type: str = "labeled",
                 mask_type: str = "grayscale",
                 palette: List = [26, 51, 77, 102, 128, 153, 179, 204, 230, 255]):#[(0, 0, 0), (128, 128, 128), (255, 255, 255)]):
        """Creates the segmentation dataset."""
        self.file_names = file_names
        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.shuffle_size = shuffle_size
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.image_dims = image_dims
        self.augment_ds = augment_ds
        self.dataset_type = dataset_type
        self.mask_type = mask_type
        self.palette = palette

    def augment(self, image, label):
        """For augmenting images and masks."""
        image, label = np.array(image), np.array(label)
        # Augmentation function
        if self.augment_ds:
            transform = A.Compose(
                [A.Flip(p=0.5),
                 A.Rotate(p=0.5),
                 A.RandomBrightnessContrast(p=0.2)])
            aug = transform(image=image, mask=label)
            return aug["image"], aug["mask"]
        else:
            return image, label

    def parse_image(self, image_path):
        """Reads and preprocesses the image."""
        image = tf.io.read_file(image_path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.cast(image, tf.float32)
        image = tf.image.resize(images=image, size=self.image_dims)
        return image

    def parse_mask(self, label_path):
        """Reads and preprocesses the mask."""
        label = tf.io.read_file(label_path)
        label = tf.image.decode_png(label, channels=3)
        label = tf.cast(label, tf.int32)
        label = tf.image.resize(images=label,
                                size=self.image_dims,
                                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        return label

    def one_hot_encode_mask(self, mask):
        """
        Converts mask to a one-hot encoding specified by the semantic map.
        """
        if self.mask_type == "grayscale":
            mask = tf.expand_dims(mask, axis=-1)
            
        one_hot_map = []
        for colour in self.palette:
            class_map = tf.reduce_all(tf.equal(mask, colour), axis=-1)
            one_hot_map.append(class_map)
        one_hot_map = tf.stack(one_hot_map, axis=-1)
        one_hot_map = tf.cast(one_hot_map, tf.float32)

        return one_hot_map

    def parse_sem_segmentation(self, file_name):
        file_name = bytes.decode(file_name, encoding="utf-8")
        image_file_path = os.path.join(self.dataset_path,
                                       self.images_dir,
                                       file_name + ".jpg")
        label_file_path = os.path.join(self.dataset_path,
                                       self.labels_dir,
                                       file_name + ".png")
        image = self.parse_image(image_file_path)
        label = self.parse_mask(label_file_path)
        image, label = self.augment(image, label)
        label = self.one_hot_encode_mask(label)
        return image, label

    def __call__(self):
        ds = tf.data.Dataset.from_tensor_slices(self.file_names)
        ds = ds.map(
            lambda x: tf.numpy_function(
                self.parse_sem_segmentation,
                inp=[x],
                Tout=[tf.float32, tf.float32]),
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
            name="object_detection_parser")
        ds = ds.shuffle(self.shuffle_size)
        ds = ds.batch(self.batch_size)
        ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
        return ds

def load_data(dataset_path, file_name="train.txt"):
    """Reads each line of the file."""
    file_names = []
    with open(
        os.path.join(
            dataset_path, file_name)) as reader:
        for line in reader.readlines():
            file_names.append(line.rstrip().split(" ")[0])
    random.shuffle(file_names)
    return file_names