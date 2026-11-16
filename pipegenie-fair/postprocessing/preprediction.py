from pipegenie.metrics import root_mean_squared_error


def select_best_mitigator(
    fair_estimators,
    X_validation,
    y_validation
):
    """
    Select the mitigated estimator with the lowest RMSE.

    Parameters
    ----------
    fair_estimators : list
        List of (name, mitigator) tuples.

    X_validation : DataFrame
    y_validation : Series

    Returns
    -------
    tuple
        (best_name, best_mitigator, best_rmse)
    """

    best_name = None
    best_mitigator = None
    best_rmse = float("inf")

    for name, mitigator in fair_estimators:

        y_pred = mitigator.predict(X_validation)

        rmse = root_mean_squared_error(
            y_validation,
            y_pred
        )

        if rmse < best_rmse:
            best_rmse = rmse
            best_name = name
            best_mitigator = mitigator

    return best_name, best_mitigator, best_rmse