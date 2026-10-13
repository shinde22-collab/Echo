import json
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from http.server import ThreadingHTTPServer
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np

from echo.data import FEATURES, ROOT, load_partitions
from echo.extraction import extract
from echo.oracle import BudgetExceeded, prediction_oracle
from echo.server import make_handler


class ProtocolTests(unittest.TestCase):
    def test_disjoint_unique_partitions(self):
        frame, victim, pool, test, metadata = load_partitions()
        self.assertEqual(metadata["unique_rows"], 1348)
        self.assertEqual(metadata["removed_duplicates"], 24)
        self.assertFalse(set(victim) & set(pool))
        self.assertFalse(set(victim) & set(test))
        self.assertFalse(set(pool) & set(test))
        self.assertEqual(len(set(victim) | set(pool) | set(test)), len(frame))
        self.assertFalse(frame.duplicated(subset=FEATURES).any())

    def test_budget_is_atomic_and_counts_rows(self):
        query, usage = prediction_oracle(lambda x: np.zeros(len(x)), 3)
        query(np.ones((2, 4)))
        with self.assertRaises(BudgetExceeded):
            query(np.ones((2, 4)))
        self.assertEqual(usage()["used"], 2)
        query(np.ones((1, 4)))
        self.assertEqual(usage()["remaining"], 0)

    def test_invalid_queries_do_not_spend_budget(self):
        query, usage = prediction_oracle(lambda x: np.zeros(len(x)), 3)
        for malformed in [[], [[1, 2, 3]], [[1, 2, 3, float("nan")]], [1, 2, 3, 4]]:
            with self.assertRaises(ValueError):
                query(malformed)
        self.assertEqual(usage()["used"], 0)

    def test_concurrent_queries_cannot_exceed_budget(self):
        query, usage = prediction_oracle(lambda x: np.zeros(len(x)), 7)
        def spend(_):
            try:
                query(np.ones((1, 4)))
                return 1
            except BudgetExceeded:
                return 0
        with ThreadPoolExecutor(max_workers=8) as executor:
            self.assertEqual(sum(executor.map(spend, range(20))), 7)
        self.assertEqual(usage()["used"], 7)

    def test_strategies_use_only_unique_queried_labels(self):
        pool = np.random.default_rng(0).normal(size=(90, 4))
        for strategy in ["random", "coverage", "uncertainty"]:
            query, usage = prediction_oracle(lambda x: (x[:, 0] > 0).astype(int), 75)
            previous = []
            for budget, model, selected, labels in extract(pool, query, strategy, 42, [25, 50, 75]):
                self.assertEqual(len(set(selected)), budget)
                self.assertEqual(usage()["used"], budget)
                self.assertEqual(selected[:len(previous)].tolist(), previous)
                np.testing.assert_array_equal(labels, (pool[selected, 0] > 0).astype(int))
                self.assertEqual(len(model.predict(pool)), len(pool))
                previous = selected.tolist()

    def test_exported_metrics_match_saved_predictions(self):
        report = json.loads((ROOT / "docs/assets/experiment.json").read_text())
        reference = np.array(report["test"]["victim"])
        truth = np.array(report["test"]["labels"])
        self.assertEqual(len(report["runs"]), 90)
        for run in report["runs"]:
            prediction = np.array(run["predictions"])
            self.assertEqual(run["queries"], run["budget"])
            self.assertEqual(len(set(run["selected"])), run["budget"])
            self.assertEqual(run["disagreements"], int(np.count_nonzero(reference != prediction)))
            self.assertAlmostEqual(run["fidelity"], float(np.mean(reference == prediction)))
            self.assertAlmostEqual(run["accuracy"], float(np.mean(truth == prediction)))
            self.assertEqual(np.sum(run["confusion"]), len(reference))

    def test_http_api_exposes_predictions_and_enforces_budget(self):
        query, usage = prediction_oracle(lambda x: (x[:, 0] > 0).astype(int), 1)
        server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(query, usage))
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        try:
            endpoint = f"http://127.0.0.1:{server.server_port}/predict"
            request = Request(endpoint, data=json.dumps({"features": [[1, 2, 3, 4]]}).encode(), headers={"Content-Type": "application/json"})
            with urlopen(request) as response:
                result = json.load(response)
            self.assertEqual(result, {"predictions": [1], "used": 1, "limit": 1, "remaining": 0})
            with self.assertRaises(HTTPError) as context:
                urlopen(request)
            self.assertEqual(context.exception.code, 429)
        finally:
            server.shutdown()
            server.server_close()
            worker.join()


if __name__ == "__main__":
    unittest.main()
