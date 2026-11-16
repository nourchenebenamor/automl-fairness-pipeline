import pandas as pd

def fairness_preprocessing(X, y, sensitive_feature, strategy="oversample"):
    if strategy not in ("oversample", "undersample"):
        raise ValueError('strategy must be "oversample" or "undersample"')

    df = (X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)).copy()
    df["_target"] = y
    df["_group"] = pd.Series(sensitive_feature).fillna("Unknown").astype(str).values

    group_sizes = df["_group"].value_counts()
    target_size = group_sizes.max() if strategy == "oversample" else group_sizes.min()
    replace = strategy == "oversample"

    balanced_df = df.groupby("_group", group_keys=False).sample(
        n=target_size, replace=replace, random_state=42
    )

    y_balanced = balanced_df.pop("_target")
    X_balanced = balanced_df.drop(columns="_group")
    return X_balanced, y_balanced