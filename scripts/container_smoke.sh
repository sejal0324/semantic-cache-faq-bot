#!/usr/bin/env sh
# Build the lite image (fake backends), start it, and check the running container end to end.
# Used by CI and locally:  sh scripts/container_smoke.sh [extra docker build args...]
set -eu
IMAGE=semcache:smoke
NAME=semcache-smoke
PORT=18080

docker build --build-arg WITH_MODELS=0 "$@" -t "$IMAGE" .
docker rm -f "$NAME" >/dev/null 2>&1 || true
docker run -d --name "$NAME" -p "$PORT:8000" \
  -e LLM_PROVIDER=fake -e EMBEDDER=fake -e ADMIN_TOKEN=smoke-token "$IMAGE" >/dev/null
trap 'docker rm -f "$NAME" >/dev/null 2>&1 || true' EXIT

echo "waiting for the container to report healthy..."
i=0
until [ "$(docker inspect -f '{{.State.Health.Status}}' "$NAME")" = "healthy" ]; do
  i=$((i + 1)); [ "$i" -gt 30 ] && { docker logs "$NAME"; echo "not healthy"; exit 1; }
  sleep 1
done

python3 - "$PORT" <<'PY'
import json, sys, urllib.request
base = f"http://127.0.0.1:{sys.argv[1]}"

def call(method, path, body=None, headers=None):
    req = urllib.request.Request(base + path, method=method, headers=headers or {},
                                 data=json.dumps(body).encode() if body is not None else None)
    if body is not None:
        req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status, r.headers, r.read().decode()
    except urllib.error.HTTPError as e:
        return e.code, e.headers, e.read().decode()

checks = []
s, h, b = call("GET", "/healthz"); checks.append(("healthz", s == 200))
s, h, b = call("GET", "/"); checks.append(("demo page + CSP", s == 200 and "default-src 'none'" in h["Content-Security-Policy"]))
ask = {"tenant_id": "bazaarnow", "question": "How do I track my order?"}
s, h, b = call("POST", "/ask", ask); checks.append(("seeded FAQ is a cache hit", s == 200 and json.loads(b)["cached"]))
new = {"tenant_id": "smoke", "question": "Do you ship to Nepal?"}
first = json.loads(call("POST", "/ask", new)[2]); second = json.loads(call("POST", "/ask", new)[2])
checks.append(("miss then hit", first["cached"] is False and second["cached"] is True))
s, h, b = call("POST", "/ask", {"tenant_id": "../etc", "question": "x"}); checks.append(("bad input rejected", s == 422))
s, h, b = call("DELETE", "/cache/bazaarnow"); checks.append(("purge needs token", s == 401))
s, h, b = call("DELETE", "/cache/bazaarnow?source_id=f01", headers={"X-Admin-Token": "smoke-token"})
checks.append(("purge with token", s == 200 and json.loads(b)["removed"] == 1))
m = json.loads(call("GET", "/metrics")[2]); checks.append(("metrics count the 3 answered asks", m["requests"] == 3 and m["hits"] == 2))
for name, ok in checks:
    print(("PASS " if ok else "FAIL ") + name)
sys.exit(0 if all(ok for _, ok in checks) else 1)
PY

uid=$(docker exec "$NAME" id -u); echo "container user id: $uid"; [ "$uid" = "1000" ]
docker exec "$NAME" sh -c 'test ! -e /home/app/src/.env && test ! -e /home/app/src/tests && test ! -e /run/secrets/pipca' \
  && echo "PASS no .env, tests or build secrets in the image"
docker image ls "$IMAGE" --format 'image size: {{.Size}}'
echo "container smoke test passed"
