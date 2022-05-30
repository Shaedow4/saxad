from tensorflow import keras
from saxad.exceptions import ShapeError
from saxad.preprocessing import dynamic_modify_tensor_shape

"""
    ae_example_config_LSTM = [
    {
        'hidden_units': 100,
        'input_shape': (4, 114),
        'return_sequences': False,
        'multiply_last_layer_output': True,
        'weights_bias': [np.load("test_weights.npy"), np.load("test_weights_2.npy"), np.load("test_bias.npy")]
    }, {
        'hidden_units': 100,
        'input_shape': (4, 114),
        'return_sequences': False,
        'multiply_last_layer_output': True,
        'weights_bias': [np.load("test_weights_ae2.npy"), np.load("test_weights_ae2_2.npy"), np.load("test_bias_ae2.npy")]
    }
]

    """


def add_trained_ae_stack(model, ae_config, window_size):
    if len(model.layers) == 0:
        model.add(
            keras.layers.LSTM(
                ae_config.get("hidden_units"), input_shape=ae_config.get("input_shape"), return_sequences=ae_config.get("return_sequences")
            )
        )
        model.layers[-1].set_weights(ae_config.get("weights_bias"))
        model.layers[-1].trainable = False
    else:
        model.add(keras.layers.LSTM(ae_config.get("hidden_units")))
        model.layers[-1].set_weights(ae_config.get("weights_bias"))
        model.layers[-1].trainable = False
    if (not ae_config.get("return_sequences")) and (
        ae_config.get("multiply_last_layer_output")
    ):
        model.add(
            keras.layers.Lambda(lambda x: dynamic_modify_tensor_shape(x, window_size))
        )
        # decoder
    if (not ae_config.get("return_sequences")) and (
        not ae_config.get("multiply_last_layer_output")
    ):
        raise ShapeError("LSTM Layer cant work with Dimensions different than 3")
    if (ae_config.get("return_sequences")) and (
        ae_config.get("multiply_last_layer_output")
    ):
        raise ShapeError(
            "Dimensions off output and input are propably not going to fit"
        )
    return model


def create_encoder_decoder_model(
    window_size,
    hidden_input_units,
    input_shape,
    return_sequences,
    multiply_last_layer_output,
    ae_configs=None,
):
    model = keras.Sequential()
    if ae_configs is None:
        model.add(
            keras.layers.LSTM(
                hidden_input_units,
                input_shape=input_shape,
                return_sequences=return_sequences,
                name="current_trained_ae",
            )
        )
    else:
        for ae_config in ae_configs:
            model = add_trained_ae_stack(model, ae_config, window_size)
        model.add(
            keras.layers.LSTM(
                hidden_input_units,
                return_sequences=return_sequences,
                name="current_trained_ae",
            )
        )
    if (not return_sequences) and (multiply_last_layer_output):
        model.add(
            keras.layers.Lambda(lambda x: dynamic_modify_tensor_shape(x, window_size))
        )
    # decoder
    if (not return_sequences) and (not multiply_last_layer_output):
        raise ShapeError("LSTM Layer cant work with Dimensions different than 3")
    if (return_sequences) and (multiply_last_layer_output):
        raise ShapeError(
            "Dimensions off output and input are propably not going to fit"
        )
    model.add(keras.layers.LSTM(input_shape[1], return_sequences=True, name="decoder"))
    return model, {
        "hidden_units": hidden_input_units,
        "input_shape": input_shape,
        "return_sequences": return_sequences,
        "multiply_last_layer_output": multiply_last_layer_output,
        "weights_bias": model.get_layer("current_trained_ae").get_weights(),
    }
