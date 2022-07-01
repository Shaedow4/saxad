from saxad.anomalyDetectionThreshold import hardcoded_threshold
import numpy as np

def test_threshold():
    x = np.array([[1.3, 1.4, 4.6], [2.3, 2.5, 5.9]])
    y = np.array([[1.9, 1.6, 8.6], [2.6, 1.7, 2.5]])
    d = 0.3
    g = hardcoded_threshold(x,y,d)
    print(g)
    assert g == [1,1]