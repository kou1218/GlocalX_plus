import pandas as pd
from hydra.utils import to_absolute_path

from .tabular_dataframe import TabularDataFrame


class adult(TabularDataFrame):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        columns = ["A" + str(i) for i in range(1, 15)]
        self.continuous_columns = ["A2", "A3", "A7", "A10", "A13", "A14"]
        self.categorical_columns = [x for x in columns if x not in self.continuous_columns]
        self.target_column = "class"

        self.data = pd.read_csv(
            to_absolute_path("data/adult.csv"),
            # sep=" ",
            # names=columns + [self.target_column],
        )
