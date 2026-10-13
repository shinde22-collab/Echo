import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from sklearn.ensemble import RandomForestClassifier

from .data import FEATURES, load_partitions
from .oracle import BudgetExceeded, prediction_oracle


def make_handler(query, usage):
    class Handler(BaseHTTPRequestHandler):
        def respond(self, code, payload):
            body = json.dumps(payload).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/status":
                return self.respond(200, usage())
            self.respond(404, {"error": "Not found"})

        def do_POST(self):
            if self.path != "/predict":
                return self.respond(404, {"error": "Not found"})
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 65536:
                    return self.respond(413, {"error": "Body must be between 1 and 65536 bytes"})
                payload = json.loads(self.rfile.read(length))
                predictions = query(payload["features"])
                self.respond(200, {"predictions": predictions.tolist(), **usage()})
            except BudgetExceeded as error:
                self.respond(429, {"error": str(error), **usage()})
            except (ValueError, KeyError, TypeError):
                self.respond(400, {"error": "Expected JSON with a finite N × 4 features matrix"})
    return Handler


def main():
    parser = argparse.ArgumentParser(description="A prediction-only banknote model API")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--budget", type=int, default=500)
    args = parser.parse_args()
    if args.budget < 1:
        parser.error("budget must be positive")
    frame, train, _, _, _ = load_partitions()
    victim = RandomForestClassifier(n_estimators=160, random_state=42, n_jobs=1)
    victim.fit(frame.loc[train, FEATURES].to_numpy(), frame.loc[train, "class"].to_numpy())
    query, usage = prediction_oracle(victim.predict, args.budget)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(query, usage))
    print(f"ECHO oracle: http://127.0.0.1:{args.port} · {args.budget} row queries", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
