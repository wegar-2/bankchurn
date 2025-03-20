import numpy as np
import pandas as pd

from sklearn.preprocessing import KBinsDiscretizer
from sklearn.linear_model import LogisticRegression


if __name__ == "__main__":

    nums = np.random.uniform(0, 100, size=10_000)
    data = pd.DataFrame(data={"X": nums, "Y": np.random.randint(1, 6, size=10_000)})
    res = data["X"].to_numpy().reshape(-1, 1)

    discretizer = KBinsDiscretizer(
        n_bins=20, strategy="quantile", encode="ordinal")
    data["X_binned"] = discretizer.fit_transform(data["X"].to_numpy().reshape(-1, 1))

    pd.crosstab(index=data["Y"], columns=data["X_binned"])
