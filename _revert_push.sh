#!/usr/bin/env bash
# Откат Dockerfile к v23 + пуш (моя правка facerestore ломала boot; реальная причина была отсутствующий Flux, теперь на месте).
cd "$(dirname "$0")" || exit 1
git config user.email >/dev/null 2>&1 || git config user.email "stalker0087708@gmail.com"
git config user.name  >/dev/null 2>&1 || git config user.name  "dashalogvinovaua-png"
echo "=== diff ==="; git --no-pager diff --stat Dockerfile
git add Dockerfile
if git diff --cached --quiet; then echo "(нечего коммитить)"; else
  git commit -m "revert Dockerfile to v23: my facerestore change broke ComfyUI boot; real cause was missing flux (now on volume)"
fi
TOKEN=$(gh auth token 2>/dev/null); [ -z "$TOKEN" ] && { echo "!! нет gh-токена"; exit 2; }
URL="https://x-access-token:${TOKEN}@github.com/dashalogvinovaua-png/keepit-serverless-worker.git"
for i in 1 2 3; do
  echo "=== push $i ==="
  git -c http.version=HTTP/1.1 push "$URL" main 2>&1 && { echo ">> PUSH OK"; break; }
  sleep 3
done
git log --oneline -1
