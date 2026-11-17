.PHONY: setup format format-check lint typecheck boundaries modules tokens test check \
        smoke eval-smoke ratchet data train eval mutation

setup:        ; uv sync && git config core.hooksPath .githooks
format:       ; uv run ruff format .
format-check: ; uv run ruff format --check .
lint:         ; uv run ruff check .
typecheck:    ; uv run pyright
boundaries:   ; uv run lint-imports
modules:      ; uv run python scripts/check_module_tests.py
tokens:       ; git ls-files -z '*.py' '*.md' '*.toml' | xargs -0 uv run python scripts/check_file_tokens.py
test:         ; uv run pytest
check:        format-check lint typecheck boundaries modules tokens test   # T13 appends: ratchet

# ML targets. They start working with the ticket in brackets.
smoke:        ; uv run r0 train --config configs/ci/smoke.toml --out /tmp/r0-smoke              # [T7]
eval-smoke:   ; uv run r0 eval --suite dev --checkpoint /tmp/r0-smoke/last.pt --out /tmp/r0-eval  # [T12]
ratchet:      ; uv run python scripts/check_ratchet.py                                          # [T13]
mutation:     ; uv run python scripts/mutate_diff.py                                            # optional, P2

# Network and real data. Only agent-train.yml (or you, locally) may run these. ADR 0004.
data:         ; uv run r0 data --suite eval/suite_v0.toml                                       # [T9]
train:        ; uv run r0 train --config $(CONFIG) --out checkpoints/$(RUN)                     # [T7]
eval:         ; uv run r0 eval --suite eval/suite_v0.toml --checkpoint checkpoints/$(RUN)/last.pt --out results/runs/$(RUN).json  # [T12]
