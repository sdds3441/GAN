import functools
import os
import cv2

from matplotlib import gridspec
import matplotlib.pylab as plt
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

def crop_center(image):
  """원본 이미지에서 정사각형의 이미지를 Crop해서 Return합니다."""
  shape = image.shape
  new_shape = min(shape[1], shape[2])
  offset_y = max(shape[1] - shape[2], 0) // 2
  offset_x = max(shape[2] - shape[1], 0) // 2
  image = tf.image.crop_to_bounding_box(
      image, offset_y, offset_x, new_shape, new_shape)
  return image

@functools.lru_cache(maxsize=None)
def load_image(image_url, image_size=(256, 256), preserve_aspect_ratio=True):
  """이미지를 load합니다. url이나 image_path를 입력해주시면 됩니다."""
  # Cache images file locally.
  try:
    image_path = tf.keras.utils.get_file(os.path.basename(image_url)[-128:], image_url)
    # Load and convert to float32 numpy array, add batch dimension, and normalize to range [0, 1].
    img = tf.io.decode_image(
        tf.io.read_file(image_path),
        channels=3, dtype=tf.float32)[tf.newaxis, ...]
  except:
    img = np.array([cv2.cvtColor(cv2.imread(image_url),cv2.COLOR_BGR2RGB)/255],dtype=np.float32)

  img = crop_center(img)
  img = tf.image.resize(img, image_size, preserve_aspect_ratio=True)
  return img

def show_n(images, titles=('',)):
  """이미지를 보여주는 함수입니다."""
  n = len(images)
  image_sizes = [image.shape[1] for image in images]
  w = (image_sizes[0] * 6) // 320
  plt.figure(figsize=(w * n, w))
  gs = gridspec.GridSpec(1, n, width_ratios=image_sizes)
  for i in range(n):
    plt.subplot(gs[i])
    plt.imshow(images[i][0], aspect='equal')
    plt.axis('off')
    plt.title(titles[i] if len(titles) > i else '')
  plt.show()


hub_handle = 'https://kaggle.com/models/google/arbitrary-image-stylization-v1/TensorFlow1/256/1'
hub_module = hub.load(hub_handle)

# images 폴더 안에 스타일을 바꿀 이미지를 넣습니다.
image_list = ['gray.jpg']

# images 폴더 안에 스타일 이미지를 넣습니다.
style_list = ['proto.jpg']

# input_image 사이즈는 굳이 안 바꿔도 될 것 같습니다.
input_image_size = 384

for final_image_size in [256, 512, 1024]:  # final images 사이즈를 결정합니다. 해당 값에 따라 결과가 확연히 달라집니다.
    for image_path in image_list:  # image_list에서 선언한 이미지들을 대상으로 합니다.
        for style_path in style_list:  # style_list에서 선언한 스타일들을 대상으로 합니다.
            content_img_size = (input_image_size, input_image_size)
            style_img_size = (final_image_size, final_image_size)  # Recommended to keep it at 256.

            # 이미지를 Load 합니다.
            content_image = load_image("images/" + image_path, style_img_size)
            style_image = load_image("images/" + style_path, content_img_size)
            style_image = tf.nn.avg_pool(style_image, ksize=[3, 3], strides=[1, 1], padding='SAME')

            # 딥러닝 Inference
            outputs = hub_module(tf.constant(content_image), tf.constant(style_image))
            stylized_image = outputs[0]

            # 원본,스타일,변환 이미지를 Matplotlib을 이용하여 보여줍니다.
            show_n([content_image, style_image, stylized_image],
                   titles=['Original content images', 'Style images', 'Stylized images(' + str(final_image_size) + ')'])

            # 최종 이미지를 outputs 폴더 안에 저장합니다.
            # outputs 폴더를 만들어주셔야 해요!
            save_image = stylized_image[0].numpy() * 255
            save_image = cv2.cvtColor(save_image, cv2.COLOR_BGR2RGB)
            cv2.imwrite("outputs/" + str(final_image_size) + "_" + style_path.split(".")[0] + "_" + image_path,
                        save_image)