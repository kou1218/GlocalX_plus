import hashlib
import os
import sys
from typing import Optional

import numpy as np
import pandas as pd


def feature_name_combiner(col, value) -> str:
    def replace(s):
        return s.replace("<", "lt_").replace(">", "gt_").replace("=", "eq_").replace("[", "lb_").replace("]", "ub_")

    col = replace(str(col))
    value = replace(str(value))
    return f'{col}="{value}"'


def feature_name_restorer(feature_name) -> str:
    return (
        feature_name.replace("lt_", "<").replace("gt_", ">").replace("eq_", "=").replace("lb_", "[").replace("ub_", "]")
    )


def calculate_md5(fpath: str, chunk_size: int = 1024 * 1024) -> str:
    # Setting the `usedforsecurity` flag does not change anything about the functionality, but indicates that we are
    # not using the MD5 checksum for cryptography. This enables its usage in restricted environments like FIPS. Without
    # it torchvision.datasets is unusable in these environments since we perform a MD5 check everywhere.
    if sys.version_info >= (3, 9):
        md5 = hashlib.md5(usedforsecurity=False)
    else:
        md5 = hashlib.md5()
    with open(fpath, "rb") as f:
        while chunk := f.read(chunk_size):
            md5.update(chunk)
    return md5.hexdigest()


def check_md5(fpath: str, md5: str, **kwargs: any) -> bool:
    return md5 == calculate_md5(fpath, **kwargs)


def check_integrity(fpath: str, md5: Optional[str] = None) -> bool:
    if not os.path.isfile(fpath):
        return False
    if md5 is None:
        return True
    return check_md5(fpath, md5)


def get_sklearn_ds(data):
    """
    This function read, split and process datasets that come from sklearn

    :param data: sklearn.utils.Bunch object that rperesents the dataset
    :param train_test_ratio: The proportion on training instances
    :return: train data and test data (pandas dataframes), names of feature columns and label column
    """
    feature_cols, label_col = data.feature_names, "class"
    X, y = data.data, data.target
    data = pd.DataFrame(X, columns=data.feature_names)
    data["class"] = y
    return data, feature_cols, label_col


def get_categorical_features(data, target_column=None):
    categorical_features_list = []
    for column in data.columns:
        data_type = data[column].dtype
        if data_type == "object" and column != target_column:
            categorical_features_list.append(column)

    return categorical_features_list


def label_encode(y: pd.Series):
    value_counts = y.value_counts(normalize=True)
    label_mapping = {value: index for index, (value, _) in enumerate(value_counts.items())}
    y_labels = y.map(label_mapping).astype(np.int32)
    return y_labels