from tensorflow import keras
from saxad import preprocessing as pre


class AttentionAutoencoder(keras.layers.Layer):
    def __init__(
        self, num_heads, key_dim, attention_encoder_output_shape, window_size, **kwargs
    ):
        super(AttentionAutoencoder, self).__init__(**kwargs)
        self.att_encoder = keras.layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=key_dim,
            output_shape=attention_encoder_output_shape,
        )
        self.normalization = keras.layers.LayerNormalization(epsilon=1e-6)
        self.return_only_last_vector_layer = keras.layers.Lambda(lambda x: x[:, -1, :])
        self.copy_last_only_vector = keras.layers.Lambda(
            lambda x: pre.dynamic_modify_tensor_shape(x, window_size)
        )

    def call(self, input):
        x = self.att_encoder(input, input)
        x = self.normalization(x)
        x = self.return_only_last_vector_layer(x)
        return self.copy_last_only_vector(x)
