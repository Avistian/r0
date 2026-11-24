#!/bin/sh
# Fail fast on checkout/index guards, then run static checks in parallel.
# Tests live in `make check`.
python3 scripts/run_check.py python scripts/check_checkout.py || exit 1
python3 scripts/run_check.py python scripts/check_tracked_files.py || exit 1
python3 scripts/run_check.py python scripts/check_artifact_sizes.py || exit 1

logs=$(mktemp -d); trap 'rm -rf "$logs"' EXIT
staged=$(git diff --cached --name-only --diff-filter=ACMR)
py=$(printf '%s\n' "$staged" | grep -E '\.py$' || true)

run() { name=$1; shift; ("$@" >"$logs/$name.log" 2>&1 || touch "$logs/$name.failed") & }

[ -n "$py" ]     && run ruff-lint   python3 scripts/run_check.py ruff check $py
[ -n "$py" ]     && run ruff-format python3 scripts/run_check.py ruff format --check $py
run tokens python3 scripts/run_check.py python scripts/check_file_tokens.py
run typecheck  python3 scripts/run_check.py pyright
run boundaries python3 scripts/run_check.py lint-imports
run modules    python3 scripts/run_check.py python scripts/check_module_tests.py
wait

status=0
for failed in "$logs"/*.failed; do
  [ -e "$failed" ] || continue
  name=$(basename "$failed" .failed); echo "✖ $name failed:"; cat "$logs/$name.log"; echo; status=1
done
exit $status
