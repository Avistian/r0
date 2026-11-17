#!/bin/sh
# scripts/pre-commit.sh: every check in parallel; budget ~3 s. Tests live in `make check`.
logs=$(mktemp -d); trap 'rm -rf "$logs"' EXIT
staged=$(git diff --cached --name-only --diff-filter=ACMR)
py=$(printf '%s\n' "$staged" | grep -E '\.py$' || true)

run() { name=$1; shift; ("$@" >"$logs/$name.log" 2>&1 || touch "$logs/$name.failed") & }

[ -n "$py" ]     && run ruff-lint   uv run ruff check $py
[ -n "$py" ]     && run ruff-format uv run ruff format --check $py
[ -n "$staged" ] && run tokens      uv run python scripts/check_file_tokens.py $staged
run typecheck  uv run pyright
run boundaries uv run lint-imports
run modules    uv run python scripts/check_module_tests.py
wait

big=$(printf '%s\n' "$staged" | while read -r f; do
  [ -f "$f" ] && [ "$(wc -c < "$f")" -gt 1048576 ] && echo "$f"; done)
[ -n "$big" ] && { echo "✖ files over 1 MB (data and checkpoints go to .cache/ or Releases): $big"; exit 1; }

status=0
for failed in "$logs"/*.failed; do
  [ -e "$failed" ] || continue
  name=$(basename "$failed" .failed); echo "✖ $name failed:"; cat "$logs/$name.log"; echo; status=1
done
exit $status
