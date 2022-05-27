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
        bool: True if output is out of the tolerance area, else False
    """
    if x * (1 + d) <= y or x * (1 - d) >= y:
        return True
    return False


via = np.vectorize(is_anomaly)


def hardcoded_threshold(
    output_model: np.array,
    output_real: np.array,
    threshold: float,
    amount_anamalous_features: int,
):
    g = via(output_real, output_model, threshold)
    for timestamp_sample in g:
        true_counter = np.unique(timestamp_sample, return_counts=True)
        true_counter_dict = {}
        for isA, amount in zip(true_counter[0], true_counter[1]):
            true_counter_dict[str(isA)] = amount
        if true_counter_dict.get("True", 0) >= math.ceil(
            len(timestamp_sample) * amount_anamalous_features
        ):
            print(
                "Timestamp is anamolous! {} features have a weird value".format(
                    true_counter_dict.get("True", 0)
                )
            )
    return g


def hmu_anomaly_score_gaussian_tail_probability_threshold(
    model_output, real_output, window_size, short_window_size
):
    raw_anomaly_scores = np.absolute(model_output - real_output)
    hmu_likelihood_scores = []
    for i in range(0, len(hmu_likelihood_scores)):
        current_window = raw_anomaly_scores[
            i + (window_size - 1) : (i + (window_size * 2 - 1))
        ]
        window_mean = np.mean(current_window)
        window_variance = np.var(current_window)
        short_current_window = raw_anomaly_scores[i : (i + short_window_size)]
        short_window_mean = np.mean(short_current_window)
        q_Function = lambda x: 0.5 - 0.5 * special.erf(
            x / np.sqrt(2)
        )  # q_function is the complementary of the error function
        hmu_likelihood_scores.append(
            1 - q_Function((short_window_mean - window_mean) / window_variance)
        )
    return hmu_likelihood_scores
