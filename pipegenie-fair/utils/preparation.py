import pandas as pd 

def prepare_data(sensitive_feature,n_groups=4,labels=None):
    return pd.qcut(sensitive_feature, q=n_groups, labels=labels)