import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.svm import SVC


def fit_surrogate(x, labels):
    if np.unique(labels).size == 1:
        return DummyClassifier(strategy="most_frequent").fit(x, labels)
    return SVC(C=10.0, kernel="rbf", gamma="scale").fit(x, labels)


def extract(pool, query, strategy, seed, budgets):
    if strategy not in {"random", "coverage", "uncertainty"}:
        raise ValueError("Unknown query strategy")
    if sorted(set(budgets)) != list(budgets) or budgets[0] < 1 or budgets[-1] > len(pool):
        raise ValueError("Budgets must be increasing, positive, and fit within the pool")
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(pool))
    selected = []
    labels = []
    available = np.ones(len(pool), dtype=bool)
    distance = np.full(len(pool), np.inf)
    model = None
    for budget in budgets:
        while len(selected) < budget:
            count = min(25, budget - len(selected))
            if strategy == "random" or not selected:
                batch = order[available[order]][:count]
            elif strategy == "uncertainty" and hasattr(model, "decision_function"):
                candidates = np.flatnonzero(available)
                uncertainty = np.abs(model.decision_function(pool[candidates]))
                batch = candidates[np.argsort(uncertainty, kind="stable")[:count]]
            elif strategy == "coverage":
                picked = []
                for _ in range(count):
                    index = int(np.argmax(np.where(available, distance, -np.inf)))
                    picked.append(index)
                    available[index] = False
                    distance = np.minimum(distance, np.sum((pool - pool[index]) ** 2, axis=1))
                batch = np.asarray(picked)
            else:
                batch = order[available[order]][:count]
            acquired = query(pool[batch])
            selected.extend(batch.tolist())
            labels.extend(acquired.tolist())
            available[batch] = False
            if strategy == "coverage":
                for index in batch:
                    distance = np.minimum(distance, np.sum((pool - pool[index]) ** 2, axis=1))
            model = fit_surrogate(pool[selected], np.asarray(labels))
        yield budget, model, np.asarray(selected), np.asarray(labels)
