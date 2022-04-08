import tensorflow as tf


def get_backbone(name='efficientnet_b0',
                 weights='imagenet',
                 segmentation=True):

    models = {
        'efficientnet_b0': tf.keras.applications.EfficientNetB0,
        'efficientnet_b1': tf.keras.applications.EfficientNetB1,
        'efficientnet_b2': tf.keras.applications.EfficientNetB2,
        'efficientnet_b3': tf.keras.applications.EfficientNetB3,
        'efficientnet_b4': tf.keras.applications.EfficientNetB4,
        'efficientnet_b5': tf.keras.applications.EfficientNetB5,
        'efficientnet_b6': tf.keras.applications.EfficientNetB6,
        'efficientnet_b7': tf.keras.applications.EfficientNetB7
    }

    backbone = models[name](include_top=False,
                            weights=weights,
                            input_shape=[None, None, 3])

    if segmentation:
        layer_names = ["block2a_expand_activation", 
                       "block3a_expand_activation", 
                       "block4a_expand_activation", 
                       "block6a_expand_activation", 
                       "top_activation"]

    outputs = [backbone.get_layer(layer_name).output for layer_name in layer_names]

    return backbone, outputs, layer_names
