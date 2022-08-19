import tensorflow as tf

from constants import IMG_SHAPE
from train.loss import IoU_loss


class UNetModel:
    def __init__(self, input_shape=(128, 128, 3)):
        self._model = self._build_model(input_shape)

    @property
    def model(self) -> tf.keras.Model:
        return self._model

    def _build_model(self, input_shape) -> tf.keras.Model:
        inputs = tf.keras.layers.Input(shape=input_shape)

        # apply Encoder
        skips = self._encoder(input_shape)(inputs)
        x = skips[-1]
        skips = reversed(skips[:-1])

        # apply Decoder and establishing the skip connections
        x = self._decoder(skips, x)

        # This is the last layers of the model
        last = tf.keras.layers.Conv2DTranspose(
          filters=20, kernel_size=3, strides=2, padding='same')  # 64x64 -> 128x128
        x = last(x)
        outputs = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid')(x)

        return tf.keras.Model(inputs=inputs, outputs=outputs)

    def _encoder(self, input_shape):
        base_model = tf.keras.applications.MobileNetV2(input_shape=input_shape, include_top=False)

        # Use the activations of these layers
        layer_names = [
            'block_1_expand_relu',  # 64x64
            'block_3_expand_relu',  # 32x32
            'block_6_expand_relu',  # 16x16
            'block_13_expand_relu',  # 8x8
            'block_16_project',  # 4x4
        ]
        base_model_outputs = [base_model.get_layer(name).output for name in layer_names]

        # Create the feature extraction model
        encoder = tf.keras.Model(inputs=base_model.input, outputs=base_model_outputs)
        encoder.trainable = True
        return encoder

    def _decoder(self, skips, encoder_output):
        decoder_stack = [
            self._upsample_block(512, 3),  # 4x4 -> 8x8
            self._upsample_block(256, 1),  # 8x8 -> 16x16
            self._upsample_block(128, 3),  # 16x16 -> 32x32
            self._upsample_block(64, 1),  # 32x32 -> 64x64
        ]

        x = encoder_output
        for block, skip in zip(decoder_stack, skips):
            x = block(x)
            concat = tf.keras.layers.Concatenate()
            x = concat([x, skip])

        return x

    def _upsample_block(self, filters, size, apply_dropout=False):
        """Upsamples an input. Conv2DTranspose => Batchnorm => Dropout => Relu
            :param: filters: number of filters
            :param: size: filter size
            :param: apply_dropout: If True, adds the dropout layer
            :return: Upsample Sequential Model
        """
        initializer = tf.random_normal_initializer(0., 0.02)
        result = tf.keras.Sequential()
        result.add(
            tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                            padding='same',
                                            kernel_initializer=initializer,
                                            use_bias=False))
        result.add(tf.keras.layers.BatchNormalization())
        if apply_dropout:
            result.add(tf.keras.layers.Dropout(0.5))
        result.add(tf.keras.layers.ReLU())
        return result


def create_model():
    model = UNetModel(IMG_SHAPE + (3,)).model
    model.compile(optimizer='adam',
                  loss=IoU_loss,
                  metrics=[])
    return model
