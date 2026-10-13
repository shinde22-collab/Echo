import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[2]
FEATURES = ["variance", "skewness", "kurtosis", "entropy"]
DATA_FILE = ROOT / "data" / "data_banknote_authentication.txt"


def load_partitions():
    frame = pd.read_csv(DATA_FILE, header=None, names=[*FEATURES, "class"])
    original_count = len(frame)
    frame = frame.drop_duplicates(subset=FEATURES).reset_index(drop=True)
    ids = np.arange(len(frame))
    development, test = train_test_split(ids, test_size=0.2, stratify=frame["class"], random_state=42)
    victim, pool = train_test_split(development, train_size=500, stratify=frame.loc[development, "class"], random_state=42)
    metadata = {
        "source_rows": original_count,
        "unique_rows": len(frame),
        "removed_duplicates": original_count - len(frame),
        "victim_rows": len(victim),
        "pool_rows": len(pool),
        "test_rows": len(test),
        "split_seed": 42,
        "sha256": hashlib.sha256(DATA_FILE.read_bytes()).hexdigest(),
        "source": "https://archive.ics.uci.edu/dataset/267/banknote+authentication",
        "citation": "Lohweg, V. (2012). Banknote Authentication. UCI Machine Learning Repository. https://doi.org/10.24432/C55P57",
        "license": "CC BY 4.0",
        "partitions": {"victim": victim.tolist(), "pool": pool.tolist(), "test": test.tolist()},
    }
    return frame, victim, pool, test, metadata
