from fairlearn.reductions import GridSearch
from fairlearn.reductions import BoundedGroupLoss
from fairlearn.reductions import SquareLoss

def fairness_postprocessing(ensemble,X_train, y_train, sensitive_features):
    fair_estimators = []

    for name, est in ensemble.estimators:

        constraint = BoundedGroupLoss(
        SquareLoss(min_val=0, max_val=1),
        upper_bound=0.1
        )


        mitigator = GridSearch(
            estimator=est,
            constraints=constraint
        )

        mitigator.fit(
            X_train,
            y_train,
            sensitive_features=sensitive_features
        )

        fair_estimators.append((name, mitigator))
        
    return fair_estimators