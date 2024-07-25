import time
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from collections import defaultdict

# xgb.XGBClassifier(use_label_encoder=False, eval_metric="mlogloss")


class SISA:
    def __init__(self, shards=5, base_model=RandomForestClassifier()) -> None:
        self.shards = shards
        self.models = [base_model for _ in range(self.shards)]

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        self.input_shards = np.array_split(x, self.shards)
        self.output_shards = np.array_split(y, self.shards)

        for i, model in enumerate(self.models):
            model.fit(self.input_shards[i], self.output_shards[i])

    def predict(self, x: np.ndarray) -> None:
        prediction_results = []
        for current_data in x:
            current_pred = 0
            final_pred = 0
            freq = defaultdict(int)

            for model in self.models:
                current_pred = model.predict([current_data])[0]
                freq[current_pred] += 1

                if freq[current_pred] > freq[final_pred]:
                    final_pred = current_pred

            prediction_results.append(final_pred)

        return np.array(prediction_results)

    def delete(self, x: np.ndarray) -> None:
        for data in x:
            for i, shard in enumerate(self.input_shards):
                delete_indices = np.where(np.all(shard == data, axis=1))[0]
                if len(delete_indices) > 0:
                    self.input_shards[i] = np.delete(
                        self.input_shards[i], delete_indices, axis=0
                    )
                    self.output_shards[i] = np.delete(
                        self.output_shards[i], delete_indices
                    )
                    self.models[i].fit(self.input_shards[i], self.output_shards[i])

    # def delete(self, x: np.ndarray) -> None:
    #     try:
    #         nsi = self.input_shards.copy()
    #         nso = self.output_shards.copy()
    #         for data in x:
    #             for i, shard in enumerate(nsi):
    #                 for j, element in enumerate(shard):
    #                     if np.array_equal(data, element):
    #                         nsi[i] = np.delete(nsi[i], j, axis=0)
    #                         nso[i] = np.delete(nso[i], j)
    #                         break
    #         for i in range(len(nsi)):
    #             if len(self.input_shards[i]) != len(nsi[i]):
    #                 self.models[i].fit(nsi[i], nso[i])
    #                 self.input_shards[i] = nsi[i]
    #                 self.output_shards[i] = nso[i]
    #     except (TypeError, ValueError) as e:
    #         print(f"Error occurred: {e}")


if __name__ == "__main__":
    testing = SISA()
    print(testing.models)
