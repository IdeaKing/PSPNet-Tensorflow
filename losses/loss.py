import tensorflow as tf
import tensorflow.keras.backend as K


def gather_channels(*xs):
    return xs


class CategoricalCELoss(tf.keras.losses.Loss):
    def __init__(self, class_weights=None):
        super().__init__(name="categorical_crossentropy")
        self.class_weights = class_weights

    def call(self, y_true, y_pred):
        y_true, y_pred = gather_channels(y_true, y_pred)

        axis = 3 if K.image_data_format() == "channels_last" else 1
        y_pred /= K.sum(y_pred, axis=axis, keepdims=True)

        y_pred = K.clip(y_pred, K.epsilon(), 1.0 - K.epsilon())

        loss = y_true * K.log(y_pred) * self.class_weights
        return - K.mean(loss)


class CategoricalFocalLoss(tf.keras.losses.Loss):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__(name="focal_loss")
        self.alpha = alpha
        self.gamma = gamma

    def call(self, y_true, y_pred):
        y_true, y_pred = gather_channels(y_true, y_pred)
        y_pred = K.clip(y_pred, K.epsilon(), 1.0 - K.epsilon())

        loss = - y_true * (self.alpha * K.pow((1 - y_pred),
                           self.gamma) * K.log(y_pred))

        return K.mean(loss)
