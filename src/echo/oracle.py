from threading import Lock

import numpy as np


class BudgetExceeded(ValueError):
    pass


def prediction_oracle(predict, budget):
    consumed = 0
    lock = Lock()

    def query(features):
        nonlocal consumed
        x = np.asarray(features, dtype=float)
        if x.ndim != 2 or x.shape[1] != 4 or len(x) == 0 or not np.isfinite(x).all():
            raise ValueError("Expected a nonempty finite matrix with four features per row")
        with lock:
            if consumed + len(x) > budget:
                raise BudgetExceeded(f"Query budget exceeded: {consumed}/{budget} used")
            labels = np.asarray(predict(x), dtype=int)
            consumed += len(x)
        return labels

    def usage():
        with lock:
            return {"used": consumed, "limit": budget, "remaining": budget - consumed}

    return query, usage
