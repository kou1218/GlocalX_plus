import logging
import os
from time import time

import numpy as np
# import optuna
import pandas as pd
from hydra.utils import to_absolute_path
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier

import src.dataset as dataset
from src.dataset import TabularDataFrame
from src.rule.rule_extraction import BaseRuleExtractor
from src.rule.rules import lore_to_glocalx
from src.glocalx import GLocalX

# from .classifier import get_classifier
# from .optuna import OptimParam
# from .utils import (
#     cal_metrics,
#     load_json,
#     save_json,
#     save_ruleset,
#     set_categories_in_rule,
#     set_seed,
# )

logger = logging.getLogger(__name__)


class ExpBase:
    def __init__(self, config):
        # set_seed(config.seed)

        self.n_splits = config.n_splits
        # self.model_name = config.model.name

        # self.model_config = config.model.params
        self.exp_config = config.exp
        self.data_config = config.data

        self.dataframe: TabularDataFrame = getattr(dataset, self.data_config.name)(seed=config.seed, **self.data_config)
        # dfs = dataframe.processed_dataframes()
        # self.categories_dict = dataframe.get_categories_dict()
        # self.train, self.test = dfs["train"], dfs["test"]
        # self.columns = dataframe.all_columns
        self.target_column = self.dataframe.target_column
        self.df = self.dataframe.data

        self.rule_extractor = BaseRuleExtractor(self.df, self.target_column)
        self.glocalx = GLocalX()

        # Onehotter for MLP in ReRx
        # if self.model_name == "rerx":
        #     all_cate = pd.concat([self.train, self.test])[dataframe.categorical_columns]
        #     self.onehoter = OneHotEncoder(sparse_output=False).fit(all_cate) if len(all_cate.columns) != 0 else None
        # else:
        #     self.onehoter = None

        self.seed = config.seed
        self.init_writer()

    def init_writer(self):
        ...

    def add_results(self, i_fold, scores: dict, time):
        ...

    def run(self):
        bb_name = "RandomForestClassifier"
        if bb_name == "RandomForestClassifier":
            bb = RandomForestClassifier(n_estimators=100, random_state=self.seed)
        elif bb_name == "DecisionTreeClassifier":
            bb = DecisionTreeClassifier(random_state=self.seed)


        self.glocalx = GLocalX( model_ai=bb)
        self.glocalx.fit(self.rule_extractor)
        alpha = 10
        global_explanations = self.glocalx.get_fine_boundary_alpha(alpha)
        print(len(global_explanations))


    def get_model_config(self, *args, **kwargs):
        raise NotImplementedError()

    def get_unique(self, train_data):
        uniq = np.unique(train_data[self.target_column])
        return uniq

    def get_x_y(self, train_data):
        x, y = train_data[self.columns], train_data[self.target_column].values.squeeze()
        return x, y


class ExpSimple(ExpBase):
    def __init__(self, config):
        super().__init__(config)

    def get_model_config(self, *args, **kwargs):
        return self.model_config
