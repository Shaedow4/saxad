import numpy as np
import math
from scipy import special


def is_anomaly(x, y, d):
    """Calculate if model output is an anomaly in one of the features

    Args:
        x (float): real output
        y (float): model output
        d (float): threshold, between 0 and 1, showing percentual deviation between real and model output

    Returns:
        bool: False if output is in tolerance area, else False
    """
    if (x <= (y * (1.0 + d))) and (x >= (y * (1.0 - d))):
        return 0
    return 1


via = np.vectorize(is_anomaly)


def hardcoded_threshold(
    output_model: np.array,
    output_real: np.array,
    threshold: float,
):
    return via(output_real, output_model, threshold)
    
    

def is_timestamp_anomal(hardcoded_threshold_result: np.array, amount_anomalous_features: int):
    """Summarize if an timestamp is anomal by adding the amount of anomal features. If it is percentual higher than 
    mount_anomalous_features return true

    hardcoded_threshold_result: output of hardcoded_threshold function from this module
    amount_anomalous_features: threshold. if more timestamps attributes are anomal than the value of the parameter, the complete timestamp is anomal
    """
    is_anomal_list = []
    amount_anomal_features_list = []
    for timestamp_sample in hardcoded_threshold_result:
        true_counter = np.unique(timestamp_sample, return_counts=1)
        anomal_features = true_counter[1][1]
        amount_anomal_features_list.append(anomal_features)
        if anomal_features >= amount_anomalous_features:
            is_anomal_list.append(1)
        else:
            is_anomal_list.append(0) 
    return is_anomal_list, amount_anomal_features_list

def hmu_anomaly_score_gaussian_tail_probability_threshold(
    model_output, real_output, window_size, short_window_size
):
    raw_anomaly_scores = np.absolute(model_output - real_output)
    hmu_likelihood_scores = []
    for i in range(window_size, len(raw_anomaly_scores)):
        current_window = raw_anomaly_scores[
            (i - i) : i
        ]
        window_mean = np.mean(current_window)
        window_variance = np.var(current_window)
        short_current_window = raw_anomaly_scores[i - short_window_size : i]
        short_window_mean = np.mean(short_current_window)
        q_Function = lambda x: 0.5 - 0.5 * special.erf(
            x / np.sqrt(2)
        )  # q_function is the complementary of the error function
        hmu_likelihood_scores.append(
            1 - q_Function((short_window_mean - window_mean) / window_variance)
        )
    return hmu_likelihood_scores
