from attr import mutable
from tensorflow import keras
from saxad import preprocessing as pre


class AttentionAutoencoder(keras.layers.Layer):
    def __init__(
        self, num_heads, key_dim, attention_encoder_output_shape, window_size, multiply_last_layer, **kwargs
    ):
        super(AttentionAutoencoder, self).__init__(**kwargs)
        self.att_encoder = keras.layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=key_dim,
            output_shape=attention_encoder_output_shape,
        )
        self.normalization = keras.layers.LayerNormalization(epsilon=1e-6)
        self.multiply_last_layer = multiply_last_layer
        if self.multiply_last_layer:
            self.return_only_last_vector_layer = keras.layers.Lambda(lambda x: x[:, -1, :])
            self.copy_last_only_vector = keras.layers.Lambda(
            lambda x: pre.dynamic_modify_tensor_shape(x, window_size)
        )

    def call(self, input):
        x = self.att_encoder(input, input)
        x = self.normalization(x)
        if self.multiply_last_layer:
            x = self.return_only_last_vector_layer(x)
            x = self.copy_last_only_vector(x)
        return x