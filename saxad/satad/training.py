from tensorflow import keras
from saxad.preprocessing import positional_encoding
from saxad.satad.attention_ae_layer import AttentionAutoencoder

"""
    ae_example_satad_config_attention = {
        "num_heads": int,
        "key_dim": int,
        "attention_encoder_output_shape: int
        "window_size": int
        "weights_bias": Union[np.array, np.array]
    } 
]
    """


def add_trained_ae_stack(ae_config, name=None):
    if name is None:
        return AttentionAutoencoder(
            ae_config.get("num_heads"),
            ae_config.get("key_dim"),
            ae_config.get("attention_encoder_output_shape"),
            ae_config.get("window_size"),
            ae_config.get("multiply_last_layer")
        )
    return AttentionAutoencoder(
        ae_config.get("num_heads"),
        ae_config.get("key_dim"),
        ae_config.get("attention_encoder_output_shape"),
        ae_config.get("window_size"),
        ae_config.get("multiply_last_layer"),
        name=name,
    )


def create_encoder_decoder_model(
    attention_heads,
    window_size,
    key_dim,
    attention_encoder_output_shape,
    input_shape,
    output_neurons,
    multiply_last_layer,
    ae_configs=None,
):
    windowed_dataset = keras.layers.Input(shape=input_shape)
    #positional_embedding = keras.layers.Lambda(lambda x: positional_encoding(x))(
    #    windowed_dataset
    #)
    weights_dict = {}
    layers_dict = {}
    if ae_configs is None:
        new_self_att_ae = AttentionAutoencoder(
            attention_heads,
            key_dim,
            attention_encoder_output_shape,
            window_size,
            multiply_last_layer,
            name="current_trained_ae"
            )(windowed_dataset)
    else:
        for index, ae_config in enumerate(ae_configs):
            if index == 0:
                layers_dict["self_att_ae_0"] = add_trained_ae_stack(
                    ae_config, "self_att_ae_0"
                )(windowed_dataset)
                weights_dict["self_att_ae_0"] = ae_config.get("weights_bias")
                continue
            g = layers_dict.get("self_att_ae_{}".format(index - 1))
            layers_dict["self_att_ae_{}".format(index)] = add_trained_ae_stack(
                ae_config, "self_att_ae_{}".format(index)
            )(g)
            weights_dict["self_att_ae_{}".format(index)] = ae_config.get("weights_bias")
        new_self_att_ae = AttentionAutoencoder(
            attention_heads,
            key_dim,
            attention_encoder_output_shape,
            window_size,
            multiply_last_layer=multiply_last_layer,
            name="current_trained_ae",
        )(layers_dict["self_att_ae_{}".format(len(ae_configs) - 1)])

    decoder_self_attention = keras.layers.MultiHeadAttention(
        num_heads=attention_heads, key_dim=key_dim, output_shape=output_neurons, dropout=0.2
    )(new_self_att_ae, new_self_att_ae)
    model = keras.models.Model(windowed_dataset, decoder_self_attention)
    for layer in model.layers:
        if layer.name not in weights_dict.keys():
            continue
        model.get_layer(layer.name).set_weights(weights_dict[layer.name])
        model.get_layer(layer.name).trainable = False

    return model, {
        "num_heads": attention_heads,
        "attention_encoder_output_shape": attention_encoder_output_shape,
        "key_dim": key_dim,
        "window_size": window_size,
        "weights_bias": model.get_layer("current_trained_ae").get_weights(),
        "multiply_last_layer": multiply_last_layer
    }
