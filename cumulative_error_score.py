import numpy as np
from typing import List


def calculate_cumulative_error_score(
    similarities: List[float],
) -> int:
    """
    Given a list of similarity values, return the cumulative error score (CES) as an
    integer.

    Args:
        similarities (List[float]): List of similarity values

    Returns:
        int: Cumulative Error Score (CES
    """
    group_counts = calculate_similarity_group_counts(similarities)
    error_scores = group_counts * np.arange(100, -1, -1)
    return np.sum(error_scores)/len(similarities)


def calculate_similarity_group_counts(
    similarities: List[float],
) -> np.ndarray:
    """
    Group similarities into bins of size bin_size. Given a list of float numbers return
    a numpy array of counts of the number of similarities that fall into each bin.

    Args:
        similarities (List[float]): list of similarity values (float)

    Returns:
        np.ndarray: number of similarity values that fall into each bin
    """
    rounded_similarities = []
    for similarity in similarities:
        if similarity < 0.0 or similarity > 1.0:
            raise ValueError("Similarity values must be between 0 and 1."
                             + f"Found: {similarity}")
        rounded_similarities.append(round(similarity, 2))
    counts = np.zeros(101)
    for similarity in rounded_similarities:
        counts[int(round(similarity * 100, 0))] += 1
    return counts
