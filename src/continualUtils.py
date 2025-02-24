def calculate_forward_transfer(baseline: list[float], result_matrix):
    """
    Calculate the forward transfer of a model.
    Args:
        accuracies: list of accuracies of the model on all tasks.
    Returns:
        forward_transfer: forward transfer of the model.
    """
    k = result_matrix.shape[0]

    for i in range(1, k):
        fwt = result_matrix[i][i] - baseline[i]

    fwt = (1/k-1)*fwt

    return fwt

def calculate_backwards_transfer(result_matrix):
    k = result_matrix.shape[0]
    for i in range(k-1):
        bwt = result_matrix[k-1][i] - result_matrix[i][i]

    bwt = (1/k-1)*bwt
    return bwt