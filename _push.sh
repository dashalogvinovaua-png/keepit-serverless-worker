#!/usr/bin/env bash
# Только пуш уже готового коммита в main (IPv4 + токен gh, 3 попытки).
cd "$(dirname "$0")" || exit 1
TOKEN=$(gh auth token 2>/dev/null)
[ -z "$TOKEN" ] && { echo "!! нет gh-токена"; exit 2; }
URL="https://x-access-token:${TOKEN}@github.com/dashalogvinovaua-png/keepit-serverless-worker.git"
echo "локальный HEAD: $(git log --oneline -1)"
for i in 1 2 3; do
  echo "=== push попытка $i ==="
  if git -c http.version=HTTP/1.1 push "$URL" main 2>&1; then
    echo ">> PUSH OK"; exit 0
  fi
  echo "(не вышло, пауза 3с)"; sleep 3
done
echo "!! push не удался за 3 попытки"; exit 1
