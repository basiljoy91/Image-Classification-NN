import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import datasets

# Load data
(training_images, training_labels), (_, _) = datasets.cifar10.load_data()
training_images = training_images / 255.0

# Correct class names (all 10 classes)
class_names = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer',
               'Dog', 'Frog', 'Horse', 'Ship', 'Truck']


for i in range(16):
    plt.subplot(4,4,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(training_images[i])
    plt.xlabel(class_names[training_labels[i][0]])
plt.show()