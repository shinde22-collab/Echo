import argparse
import json
import platform
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler

from .data import FEATURES, ROOT, load_partitions
from .extraction import extract
from .oracle import prediction_oracle

BUDGETS = [25, 50, 100, 150, 250, 500]
SEEDS = [7, 19, 42, 73, 101]
STRATEGIES = ["random", "coverage", "uncertainty"]


def export_model(model):
    if not hasattr(model, "support_vectors_"):
        return {"constant": int(model.classes_[0])}
    return {
        "vectors": model.support_vectors_.tolist(),
        "dual": model.dual_coef_[0].tolist(),
        "intercept": float(model.intercept_[0]),
        "gamma": float(model._gamma),
    }


def run(output=ROOT / "results", web=ROOT / "docs" / "assets"):
    output, web = Path(output), Path(web)
    output.mkdir(parents=True, exist_ok=True)
    web.mkdir(parents=True, exist_ok=True)
    frame, train_ids, pool_ids, test_ids, metadata = load_partitions()
    x = frame[FEATURES].to_numpy(dtype=float)
    y = frame["class"].to_numpy(dtype=int)
    victim = RandomForestClassifier(n_estimators=160, min_samples_leaf=1, random_state=42, n_jobs=1)
    victim.fit(x[train_ids], y[train_ids])
    scaler = StandardScaler().fit(x[pool_ids])
    pool, test = scaler.transform(x[pool_ids]), scaler.transform(x[test_ids])
    reference = victim.predict(x[test_ids])
    axis_bounds = np.quantile(x[pool_ids], [0.01, 0.99], axis=0).T
    width, height = 80, 60
    gx, gy = np.meshgrid(np.linspace(*axis_bounds[0], width), np.linspace(*axis_bounds[1], height))
    grid = np.tile(np.median(x[pool_ids], axis=0), (width * height, 1))
    grid[:, 0], grid[:, 1] = gx.ravel(), gy.ravel()
    victim_grid = victim.predict(grid)
    report = {
        "protocol": "echo-v1",
        "dataset": metadata,
        "features": FEATURES,
        "budgets": BUDGETS,
        "seeds": SEEDS,
        "strategies": STRATEGIES,
        "victim": {"type": "RandomForestClassifier", "trees": 160, "seed": 42, "accuracy": float(accuracy_score(y[test_ids], reference))},
        "surrogate": {"type": "RBF SVC", "C": 10, "gamma": "scale", "labels": "oracle hard labels only"},
        "environment": {"python": platform.python_version(), "sklearn": sklearn.__version__, "numpy": np.__version__, "pandas": pd.__version__, "matplotlib": matplotlib.__version__},
        "scaler": {"mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist()},
        "test": {"x": x[test_ids].tolist(), "labels": y[test_ids].tolist(), "victim": reference.tolist(), "ids": test_ids.tolist()},
        "pool": {"x": x[pool_ids].tolist(), "ids": pool_ids.tolist()},
        "slice": {"width": width, "height": height, "bounds": axis_bounds.tolist(), "fixed": np.median(x[pool_ids], axis=0).tolist(), "victim": "".join(map(str, victim_grid))},
        "runs": [],
    }
    metrics = []
    for strategy in STRATEGIES:
        for seed in SEEDS:
            query, usage = prediction_oracle(lambda z: victim.predict(scaler.inverse_transform(z)), max(BUDGETS))
            for budget, model, selected, labels in extract(pool, query, strategy, seed, BUDGETS):
                predictions = model.predict(test)
                row = {
                    "strategy": strategy,
                    "seed": seed,
                    "budget": budget,
                    "fidelity": float(accuracy_score(reference, predictions)),
                    "accuracy": float(accuracy_score(y[test_ids], predictions)),
                    "f1": float(f1_score(y[test_ids], predictions, zero_division=0)),
                    "disagreements": int(np.count_nonzero(reference != predictions)),
                    "queries": usage()["used"],
                }
                metrics.append(row)
                report["runs"].append({
                    **row,
                    "selected": selected.tolist(),
                    "acquired_labels": labels.tolist(),
                    "predictions": predictions.tolist(),
                    "confusion": confusion_matrix(reference, predictions, labels=[0, 1]).tolist(),
                    "model": export_model(model),
                    "slice": "".join(map(str, model.predict(scaler.transform(grid)))),
                })
            print(f"{strategy:12s} seed {seed:3d}: {metrics[-1]['fidelity']:.2%} fidelity at 500 queries", flush=True)
    table = pd.DataFrame(metrics)
    table.to_csv(output / "metrics.csv", index=False, float_format="%.12f")
    summary = []
    for strategy in STRATEGIES:
        for budget in BUDGETS:
            values = table[(table.strategy == strategy) & (table.budget == budget)]
            summary.append({"strategy": strategy, "budget": budget, "mean": float(values.fidelity.mean()), "std": float(values.fidelity.std(ddof=1)), "min": float(values.fidelity.min()), "max": float(values.fidelity.max())})
    report["summary"] = summary
    payload = json.dumps(report, separators=(",", ":"), allow_nan=False)
    (web / "experiment.json").write_text(payload, encoding="utf-8")
    (output / "summary.json").write_text(json.dumps({k: v for k, v in report.items() if k not in {"runs", "pool", "test", "slice", "scaler"}}, indent=2), encoding="utf-8")
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(10, 5.6), facecolor="#0b1110")
    ax.set_facecolor("#0b1110")
    for strategy, color in zip(STRATEGIES, ["#8cabff", "#e8b17d", "#d5ff70"]):
        rows = [r for r in summary if r["strategy"] == strategy]
        means = np.array([r["mean"] for r in rows]) * 100
        sd = np.array([r["std"] for r in rows]) * 100
        ax.plot(BUDGETS, means, "o-", color=color, label=strategy.title(), linewidth=2)
        ax.fill_between(BUDGETS, means - sd, means + sd, color=color, alpha=0.13)
    ax.set(xlabel="Oracle queries", ylabel="Held-out prediction agreement (%)", title="ECHO / Black-box extraction fidelity")
    ax.grid(alpha=0.1)
    ax.legend(frameon=False)
    fig.text(0.13, 0.015, f"Mean ± 1 sample SD across 5 acquisition seeds · fixed victim and split · n={len(test_ids)} test rows", color="#9ba89e", fontsize=9)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(output / "fidelity.png", dpi=180, facecolor=fig.get_facecolor())
    plt.close(fig)
    (web / "metrics.csv").write_bytes((output / "metrics.csv").read_bytes())
    print(f"Exported {len(metrics)} measured checkpoints; {metadata['removed_duplicates']} duplicate feature rows removed.")
    return report


def main():
    parser = argparse.ArgumentParser(description="Run the reproducible ECHO extraction benchmark")
    parser.add_argument("--output", type=Path, default=ROOT / "results")
    parser.add_argument("--web", type=Path, default=ROOT / "docs" / "assets")
    args = parser.parse_args()
    run(args.output, args.web)


if __name__ == "__main__":
    main()

