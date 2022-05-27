from saxad.preprocessing import positional_encoding
import tensorflow
import numpy as np


def test_pos_encoding():
    x = np.array([[[2.0]]])
    g = positional_encoding(x)
    print(g)
    assert g == [[[0]]]
