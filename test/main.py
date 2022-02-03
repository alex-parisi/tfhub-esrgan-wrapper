import os
import time
from PIL import Image
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt
os.environ["TFHUB_DOWNLOAD_PROGRESS"] = "True"

# Declaring Constants
IMAGE_PATH = "test.jpg"
SAVED_MODEL_PATH = "https://tfhub.dev/captain-pool/esrgan-tf2/1"


def gridify_tensor(tensor, grid_size=128):
    zero_pad_height = tensor.shape[1] % grid_size
    zero_pad_width = tensor.shape[2] % grid_size

    paddings = tf.constant([[0, 0], [0, grid_size - zero_pad_height], [0, grid_size - zero_pad_width], [0, 0]])
    tensor = tf.pad(tensor, paddings, "CONSTANT")
    tensor_grid = []
    I = tensor.shape[1] // grid_size
    J = tensor.shape[2] // grid_size
    for i in range(I):
        for j in range(J):
            tensor_slice = tensor[:, (i*grid_size):(i*grid_size)+grid_size, (j*grid_size):(j*grid_size)+grid_size, :]
            tensor_grid.append(tensor_slice)

    return tensor_grid, (I, J)


def tensorfy_grid(grid, grid_structure, original_tensor_shape):
    tensor_rows = []
    for i in range(grid_structure[0]):
        tensor_row = tf.zeros(grid[0].shape)
        for j in range(grid_structure[1]):
            current_tensor = grid[i*grid_structure[1]+j]
            if j == 0:
                tensor_row = current_tensor
            else:
                tensor_row = tf.concat([tensor_row, current_tensor], 2)
        tensor_rows.append(tensor_row)
    tensor = tf.concat(tensor_rows, 1)

    return tensor[:, 0:original_tensor_shape[1], 0:original_tensor_shape[2], :]


def preprocess_image(image_path):
    hr_image = tf.image.decode_image(tf.io.read_file(image_path))
    # If PNG, remove the alpha channel. The model only supports
    # images with 3 color channels.
    if hr_image.shape[-1] == 4:
        hr_image = hr_image[...,:-1]
    hr_size = (tf.convert_to_tensor(hr_image.shape[:-1]) // 4) * 4
    hr_image = tf.image.crop_to_bounding_box(hr_image, 0, 0, hr_size[0], hr_size[1])
    hr_image = tf.cast(hr_image, tf.float32)
    return tf.expand_dims(hr_image, 0)


def save_image(image, filename):
    if not isinstance(image, Image.Image):
        image = tf.clip_by_value(image, 0, 255)
        image = Image.fromarray(tf.cast(image, tf.uint8).numpy())
    image.save("%s.jpg" % filename)
    print("Saved as %s.jpg" % filename)


def plot_image(image, title=""):
    image = np.asarray(image)
    image = tf.clip_by_value(image, 0, 255)
    image = Image.fromarray(tf.cast(image, tf.uint8).numpy())
    plt.imshow(image)
    plt.axis("off")
    plt.title(title)
    plt.show()


model = hub.load(SAVED_MODEL_PATH)

hr_image = preprocess_image(IMAGE_PATH)
original_shape = hr_image.shape

tensor_grid, grid_structure = gridify_tensor(hr_image)

high_res_grid = []
for i in range(len(tensor_grid)):
    print(f"{i}, {len(tensor_grid)}")
    tensor = tensor_grid[i]
    high_res_tensor = model(tensor)
    high_res_grid.append(high_res_tensor)

high_res_tensor = tensorfy_grid(high_res_grid, grid_structure, tuple([4*x for x in original_shape]))

plot_image(tf.squeeze(hr_image))
save_image(tf.squeeze(hr_image), "input")
plot_image(tf.squeeze(high_res_tensor))
save_image(tf.squeeze(high_res_tensor), "output")










# # Plotting Original Resolution image
# plot_image(tf.squeeze(hr_image), title="Original Image")
# save_image(tf.squeeze(hr_image), filename="Original Image")
#
# model = hub.load(SAVED_MODEL_PATH)
#
# start = time.time()
# fake_image = model(hr_image)
# fake_image = tf.squeeze(fake_image)
# print("Time Taken: %f" % (time.time() - start))
#
# # Plotting Super Resolution Image
# plot_image(tf.squeeze(fake_image), title="Super Resolution")
# save_image(tf.squeeze(fake_image), filename="Super Resolution")