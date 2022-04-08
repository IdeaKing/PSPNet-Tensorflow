import tensorflow as tf
import tensorflow.keras.backend as K


class ConvolutionBnActivation(tf.keras.layers.Layer):
    def __init__(self, 
                 filters, 
                 kernel_size, 
                 strides=(1, 1), 
                 padding="same", 
                 data_format=None, 
                 dilation_rate=(1, 1),
                 activation=None, 
                 kernel_initializer="glorot_uniform", 
                 bias_initializer="zeros", 
                 kernel_regularizer=None,
                 bias_regularizer=None, 
                 activity_regularizer=None, 
                 kernel_constraint=None, 
                 bias_constraint=None, 
                 use_batchnorm=False, 
                 axis=-1, 
                 momentum=0.99, 
                 epsilon=0.001, 
                 center=True, 
                 scale=True, 
                 trainable=True,
                 post_activation="relu", 
                 block_name=None):
        super(ConvolutionBnActivation, self).__init__()


        # 2D Convolution Arguments
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.data_format = data_format
        self.dilation_rate = dilation_rate
        self.activation = activation
        self.use_bias = not (use_batchnorm)
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer
        self.activity_regularizer = activity_regularizer
        self.kernel_constraint = kernel_constraint
        self.bias_constraint = bias_constraint

        # Batch Normalization Arguments
        self.axis = axis
        self.momentum = momentum
        self.epsilon = epsilon
        self.center = center
        self.scale = scale
        self.trainable = trainable
        
        self.block_name = block_name
        
        self.conv = None
        self.bn = None
        self.post_activation = tf.keras.layers.Activation(post_activation)

    def build(self, input_shape):
        self.conv = tf.keras.layers.Conv2D(
            filters=self.filters,
            kernel_size=self.kernel_size,
            strides=self.strides,
            padding=self.padding,
            data_format=self.data_format,
            dilation_rate=self.dilation_rate,
            activation=self.activation,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            activity_regularizer=self.activity_regularizer,
            kernel_constraint=self.kernel_constraint,
            bias_constraint=self.bias_constraint,
            name=self.block_name + "_conv" if self.block_name is not None else None

        )

        self.bn = tf.keras.layers.BatchNormalization(
            axis=self.axis,
            momentum=self.momentum,
            epsilon=self.epsilon,
            center=self.center,
            scale=self.scale,
            trainable=self.trainable,
            name=self.block_name + "_bn" if self.block_name is not None else None
        )

    def call(self, x, training=None):
        x = self.conv(x)
        x = self.bn(x, training=training)
        x = self.post_activation(x)

        return x

    def compute_output_shape(self, input_shape):
        print(input_shape)
        return [input_shape[0], input_shape[1], input_shape[2], self.filters]

class SpatialContextBlock(tf.keras.layers.Layer):
    def __init__(self, level, filters=256, pooling_type="avg"):
        super(SpatialContextBlock, self).__init__()

        self.level = level
        self.filters = filters
        self.pooling_type = pooling_type

        self.pooling2d = None
        self.conv1x1_bn_relu = ConvolutionBnActivation(filters, kernel_size=(1, 1))
        self.upsample2d = None

    def build(self, input_shape):
        if self.pooling_type not in ("max", "avg"):
            raise ValueError("Unsupported pooling type - '{}'".format(self.pooling_type) + "Use 'avg' or 'max'")

        self.pooling2d = tf.keras.layers.MaxPool2D if self.pooling_type == "max" else tf.keras.layers.AveragePooling2D

        spatial_size = input_shape[1:3] if K.image_data_format() == "channels_last" else input_shape[2:]

        pool_size = up_size = [spatial_size[0] // self.level, spatial_size[1] // self.level]
        self.pooling2d = self.pooling2d(pool_size, strides=pool_size, padding="same")
        
        self.upsample2d = tf.keras.layers.UpSampling2D(up_size, interpolation="bilinear")

    def call(self, x, training=None):
        x = self.pooling2d(x, training=training)
        x = self.conv1x1_bn_relu(x, training=training)
        x = self.upsample2d(x)
        return x

    def compute_output_shape(self, input_shape):
        return input_shape