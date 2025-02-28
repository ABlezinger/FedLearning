def calculate_forward_transfer(baseline: list[float], result_matrix):
    """
    Calculate the forward transfer of a model.
    
    Forward transfer measures how learning on previous tasks affects the performance on future tasks
    before the model has been trained on them. A positive value indicates positive knowledge transfer.
    
    Args:
        baseline: list of accuracies representing initial performance on each task without any prior training
        result_matrix: matrix where result_matrix[i][j] represents the accuracy on task j after training on task i
    
    Returns:
        fwt: forward transfer value, averaged across tasks
    """
    k = result_matrix.shape[0]  # Number of tasks
    for i in range(1, k):
        # Calculate difference between performance on current task and baseline performance
        fwt = result_matrix[i][i] - baseline[i]
        # Average across all tasks (excluding the first task)
        fwt = (1/(k-1))*fwt
    return fwt


def calculate_backwards_transfer(result_matrix):
    """
    Calculate the backward transfer of a model.
    
    Backward transfer measures how learning new tasks affects the performance on previously learned tasks.
    Positive values indicate that learning new tasks improved performance on older tasks.
    
    Args:
        result_matrix: matrix where result_matrix[i][j] represents the accuracy on task j after training on task i
    
    Returns:
        bwt: backward transfer value, averaged across tasks
    """
    k = result_matrix.shape[0]  # Number of tasks
    for i in range(k-1):
        # Calculate difference between final performance on previous task and performance right after learning that task
        bwt = result_matrix[k-1][i] - result_matrix[i][i]
        # Average across all tasks (excluding the last task)
        bwt = (1/(k-1))*bwt
    return bwt