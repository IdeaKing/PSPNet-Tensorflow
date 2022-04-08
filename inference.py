import tensorflow as tf
import matplotlib.pyplot as plt

def parse_image(image_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.cast(image, tf.float32)
    image = tf.image.resize(images=image, size=(120, 120))
    return image

def create_mask(pred_mask):
    pred_mask = tf.argmax(pred_mask, axis=-1)
    pred_mask = pred_mask[..., tf.newaxis]
    return pred_mask[0]

def run_inference(image_path, model):
    image = parse_image(image_path)
    prediction = model(tf.expand_dims(image, axis=0), 
                       training=False)
    mask = create_mask(prediction)
    display_image(tf.cast(image, tf.int32), mask)
    return mask

def display_image(image, mask):
    fig = plt.figure(figsize=(20, 20))
    fig.add_subplot(1, 2, 1)
    plt.imshow(image)
    fig.add_subplot(1, 2, 2)
    plt.imshow(mask)
    plt.show()

if __name__ == "__main__":
    model = tf.keras.models.load_model(
        "training_dir/refuge/model-exported")
    run_inference(
        "data/datasets/cropped_refuge/images/g0001.jpg", 
        model)
