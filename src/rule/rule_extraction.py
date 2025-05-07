import warnings
import random
import json
import pickle
from tqdm import tqdm

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score

from lore_explainer.util import neuclidean
from lore_explainer.datamanager import prepare_adult_dataset, prepare_dataset
from lore_explainer.lorem import LOREM


class BaseRuleExtractor(object):
    """
    Base class for rule extractor
    """

    def __init__(self, df, class_name, num_samples=None, model=None, ):
        self.df = df

        self.class_name = class_name
        self.model = model
        self.num_samples = num_samples

    def extract_rules(self):
        # 使用するデータは後ほど修正する
        df, class_name = prepare_adult_dataset("/home/pc002/lab/GlocalX_plus/data/adult.csv")
        df, feature_names, class_values, numeric_columns, df_orig, real_feature_names, features_map = prepare_dataset(df, class_name)

        X_train, X_test, Y_train, Y_test = train_test_split(df[feature_names].values, df[class_name].values,
                                                            test_size=0.2,
                                                            random_state=42, 
                                                            stratify=df[class_name].values
                                            )
        bb = RandomForestClassifier(n_estimators=100, random_state=42)
        model_name = "RandomForestClassifier"
        bb.fit(X_train, Y_train)
        Y_pred = bb.predict(X_test)

        # テストデータからランダムに30個のデータを取得
        indices = random.sample(range(len(X_test)), 30)
 
        x_to_exp = X_test[indices]

        _, X_test_orig, _, _ = train_test_split(df_orig[real_feature_names].values, df_orig[class_name].values, 
                                    test_size=0.2,
                                    random_state=42, 
                                    stratify=df[class_name].values)


        # LORE initialize
        lore_explainer = LOREM(X_test_orig, bb.predict,
                                feature_names, class_name, class_values, numeric_columns, features_map,
                                neigh_type='geneticp',
                                categorical_use_prob=True,
                                continuous_fun_estimation=False,
                                size=1000, ocr=0.1, random_state=42,
                                ngen=10, bb_predict_proba=bb.predict_proba, 
                                verbose=False)
        # local ruleの作成（rangeの値の数だけ作成される）
        rules = []
        for i in tqdm(range(30)):
            exp = lore_explainer.explain_instance(x_to_exp[i], samples=50, use_weights=True, metric=neuclidean)
            rules.append(exp.rule)
        
        data = df.to_numpy()
        data = [list(sample) for sample in data]
        data = np.array(data)
        return rules, feature_names, class_values, data, bb