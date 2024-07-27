import time
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from collections import defaultdict


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

    def delete(self, x: np.ndarray) -> float:
        shard_map = {i: set() for i in range(self.shards)}

        # Mengumpulkan indeks yang akan dihapus
        for i, shard in enumerate(self.input_shards):
            for data in x:
                delete_indices = np.where(np.all(shard == data, axis=1))[0]
                if delete_indices.size > 0:
                    shard_map[i].update(delete_indices)

        # Hapus data dan retrain model jika ada perubahan
        start_time = time.time()
        for i, indices in shard_map.items():
            if indices:
                self.input_shards[i] = np.delete(
                    self.input_shards[i], list(indices), axis=0
                )
                self.output_shards[i] = np.delete(self.output_shards[i], list(indices))

                # Retrain the model for this shard
                self.models[i].fit(self.input_shards[i], self.output_shards[i])

        end_time = time.time()
        return end_time - start_time


if __name__ == "__main__":
    testing = SISA()
    print(testing.models)
