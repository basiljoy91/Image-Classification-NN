import cv2 as cv
import numpy as np
import tensorflow.keras import datasets, layers, models

(training_images, training_labels), (testing_images, testing_labels) = datasets.cifar10.loads_data()
training_images,  testing_images = testing_images / 255, testing_images / 255

class _names = ['']