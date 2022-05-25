# SAXAD

**S**tacked **A**utoencoders **X**-NeuralNetworkArchitecture **A**nomaly **D**etection

Package with training and creation functionality for **SALAD** (Stacked Autoencoders LSTM Anomaly Detection) and SATAD (currently in progress), the pendant using Transformers (Vaswani et. al 2017). After (Sagheeb et. 2019) the training consists of one autoencoder at time, which you stack up. Furthermore only the last output of a LSTM Cell is used for autoencoding. To maintain the needed input structure for a lstm cell the last output vector is copied and tiled up as a 3D Matrix.  

Furthermore the package consists of typical anomaly detection thresholds by a hardcoded threshold and with the approach of Ahmad and Purdy 2016 by calculating raw Anomaly Scores and calculate the likelihood of the scores being anomalous.

The package depends on *keras* with *tensorflow* backend. Other backends should work too, but are only tested with *tensorflow*.  

## Installation

To install the package with pip, download the repository, open a command line interface in the main diretory of this project and write

```bash
python -m pip install -e ./
```

### import

After installation, you can import the modules by calling saxad

*example*

```python
from saxad.anomalyDetectionThreshold import hmu_anomaly_score_gaussian_tail_probability_threshold
```

## To-Do
- [ ] Generalize Code, f.e training is only done with *mse*
- [ ] Documentation
- [ ] Error Handling
- [ ] Full Support for tensors and pandas.DataFrame
- [ ] Create SATAD
- [ ] Evaluate Code on more Datasets
- [ ] Make pretrained models available

Published with *MIT License*
