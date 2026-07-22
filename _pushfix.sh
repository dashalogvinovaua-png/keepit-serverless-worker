#!/usr/bin/env bash
# Коммит закалённого Dockerfile + пуш в main через токен gh → триггер авто-сборки в RunPod.
# БЕЗ set -e: хотим видеть все ошибки, а не тихо падать.
cd "$(dirname "$0")" || { echo "cd fail"; exit 1; }

git config user.email >/dev/null 2>&1 || git config user.email "stalker0087708@gmail.com"
git config user.name  >/dev/null 2>&1 || git config user.name  "dashalogvinovaua-png"

echo "=== status до ==="
git status --short

git add Dockerfile _pushfix.sh
if git diff --cached --quiet; then
  echo "(нечего коммитить — возможно уже закоммичено)"
else
  git commit -m "harden facerestore_cf: skip basicsr requirements that crash ComfyUI boot" && echo ">> commit OK"
fi

echo "=== push через gh-токен ==="
TOKEN=$(gh auth token 2>/dev/null)
if [ -z "$TOKEN" ]; then echo "!! нет gh-токена (gh auth token пуст)"; exit 2; fi
git push "https://x-access-token:${TOKEN}@github.com/dashalogvinovaua-png/keepit-serverless-worker.git" main 2>&1

echo "=== итог: локальный HEAD ==="
git log --oneline -1
