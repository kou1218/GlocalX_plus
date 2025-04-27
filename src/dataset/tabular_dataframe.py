import logging
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (
    MinMaxScaler,
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
)

from .utils import check_integrity, feature_name_combiner, label_encode

logger = logging.getLogger(__name__)


class TabularDataFrame(object):
    """Base class for datasets"""

    def __init__(
        self,
        seed,
        data_path: str,
        target_column: str,
        categorical_encoder="ordinal",
        continuous_encoder: str = None,
        test_size: float = 0.2,
        download: bool = False,
        **kwargs,
    ) -> None:
        """
        Args:
            root (str): Path to the root of datasets for saving/loading.
            download (bool): If True, you must implement `self.download` method
                in the child class. Defaults to False.
        """
        self.seed = seed
        self.data_path = data_path
        self.target_column = target_column
        self.categorical_encoder = categorical_encoder
        self.continuous_encoder = continuous_encoder
        self.test_size = test_size
        if download:
            self.download()

    @property
    def mirrors(self) -> None:
        pass

    @property
    def resources(self) -> None:
        pass

    @property
    def raw_folder(self) -> str:
        """The folder where raw data will be stored"""
        return os.path.join(self.root, self.__class__.__name__, "raw")

    def _check_exists(self, fpath: Sequence[Tuple[str, Union[str, None]]]) -> bool:
        """
        Args:
            fpath (sequence of tuple[str, (str or None)]): Each value has the format
                [file_path, (md5sum or None)]. Checking if files are correctly
                stored in `self.raw_folder`. If `md5sum` is provided, checking
                the file itself is valid.
        """

        return all(check_integrity(os.path.join(self.raw_folder, path[0]), md5=path[1]) for path in fpath)

    def download(self) -> None:
        """
        Implement this function if the dataset is downloadable.
        See :func:`~deep_table.data.datasets.adult.Adult` for an example implementation.
        """
        raise NotImplementedError


    def cat_cardinalities(self, use_unk: bool = True) -> Optional[List[int]]:
        """List of the numbers of the categories of each column.

        Args:
            use_unk (bool): If True, each feature (column) has "unknown" categories.

        Returns:
            List[int], optional: List of cardinalities. i-th value denotes
                the number of categories which i-th column has.
        """
        cardinalities = []
        df_train = self.get_dataframe(train=True)
        df_train_cat = df_train[self.categorical_columns]
        cardinalities = df_train_cat.nunique().values.astype(int)
        if use_unk:
            cardinalities += 1
        cardinalities_list = cardinalities.tolist()
        return cardinalities_list

    def show_data_details(self, train: pd.DataFrame, test: pd.DataFrame):
        all_data = pd.concat([train, test])
        logger.info(f"Dataset size       : {len(all_data)}")
        logger.info(f"All columns        : {all_data.shape[1] - 1}")
        logger.info(f"Num of cate columns: {len(self.categorical_columns)}")
        logger.info(f"Num of cont columns: {len(self.continuous_columns)}")

        y = all_data[self.target_column]
        class_ratios = y.value_counts(normalize=True)
        for label, class_ratio in zip(class_ratios.index, class_ratios.values):
            logger.info(f"class {label:<13}: {class_ratio:.3f}")

    def get_dataframe(self) -> Dict[str, pd.DataFrame]:
        self.data = pd.read_csv(self.data_path)
        self.data[self.target_column] = label_encode(self.data[self.target_column])
        train, test = train_test_split(
            self.data,
            test_size=self.test_size,
            random_state=self.seed,
            stratify=self.data[self.target_column],
        )

        self.show_data_details(train, test)
        dfs = {
            "train": train,
            "test": test,
        }
        return dfs

    def fit_feature_encoder(self, df_train):
        # Categorical values are fitted on all data.
        if self.categorical_columns != []:
            if self.categorical_encoder == "ordinal":
                self._categorical_encoder = OrdinalEncoder(dtype=np.int32).fit(self.data[self.categorical_columns])
            elif self.categorical_encoder == "onehot":
                self._categorical_encoder = OneHotEncoder(
                    sparse_output=False,
                    feature_name_combiner=feature_name_combiner,
                    dtype=np.int32,
                ).fit(self.data[self.categorical_columns])
            else:
                raise ValueError(self.categorical_encoder)
        if self.continuous_columns != [] and self.continuous_encoder is not None:
            if self.continuous_encoder == "standard":
                self._continuous_encoder = StandardScaler()
            elif self.continuous_encoder == "minmax":
                self._continuous_encoder = MinMaxScaler()
            else:
                raise ValueError(self.continuous_encoder)
            self._continuous_encoder.fit(df_train[self.continuous_columns])

    def apply_onehot_encoding(self, df: pd.DataFrame):
        encoded = self._categorical_encoder.transform(df[self.categorical_columns])
        encoded_columns = self._categorical_encoder.get_feature_names_out(self.categorical_columns)
        encoded_df = pd.DataFrame(encoded, columns=encoded_columns, index=df.index)
        df = df.drop(self.categorical_columns, axis=1)
        return pd.concat([df, encoded_df], axis=1)

    def apply_feature_encoding(self, dfs):
        for key in dfs.keys():
            if self.categorical_columns != []:
                if isinstance(self._categorical_encoder, OrdinalEncoder):
                    dfs[key][self.categorical_columns] = self._categorical_encoder.transform(
                        dfs[key][self.categorical_columns]
                    )
                else:
                    dfs[key] = self.apply_onehot_encoding(dfs[key])
            if self.continuous_columns != []:
                if self.continuous_encoder is not None:
                    dfs[key][self.continuous_columns] = self._continuous_encoder.transform(
                        dfs[key][self.continuous_columns]
                    )
                else:
                    dfs[key][self.continuous_columns] = dfs[key][self.continuous_columns].astype(np.float64)
        if self.categorical_columns != []:
            if isinstance(self._categorical_encoder, OneHotEncoder):
                self.categorical_columns = self._categorical_encoder.get_feature_names_out(self.categorical_columns)
        return dfs

    def processed_dataframes(self) -> Dict[str, pd.DataFrame]:
        """
        Returns:
            dict[str, DataFrame]: The value has the keys "train", "val" and "test".
        """
        dfs = self.get_dataframe()

        # self._init_checker()
        # dfs = self.get_classify_dataframe()
        # # preprocessing
        # self.fit_feature_encoder(dfs["train"])
        # dfs = self.apply_feature_encoding(dfs)
        # self.all_columns = list(self.categorical_columns) + list(self.continuous_columns)


        return dfs

    def num_categories(self, use_unk: bool = True) -> int:
        """Total numbers of categories

        Args:
            use_unk (bool): If True, the returned value is calculated
                as there are unknown categories.
        """
        return sum(self.cat_cardinalities(use_unk=use_unk))

    def get_categories_dict(self):
        if not hasattr(self, "_categorical_encoder"):
            return None

        categories_dict: Dict[str, List[Any]] = {}
        for categorical_column, categories in zip(self.categorical_columns, self._categorical_encoder.categories_):
            categories_dict[categorical_column] = categories.tolist()

        return categories_dict