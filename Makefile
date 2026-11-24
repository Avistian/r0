.PHONY: setup format format-check lint typecheck boundaries modules tokens test check preflight tracked-files hooks artifact-sizes \
        smoke eval-smoke ratchet data train eval mutation

setup:        ; UV_PROJECT_ENVIRONMENT=.venv uv sync && $(MAKE) hooks
hooks:        ; git config --local core.hooksPath .githooks && chmod +x .githooks/pre-commit
VERIFY = python3 scripts/run_check.py

format:       ; $(VERIFY) ruff format .
format-check: ; $(VERIFY) ruff format --check .
lint:         ; $(VERIFY) ruff check .
typecheck:    ; $(VERIFY) pyright
boundaries:   ; $(VERIFY) lint-imports
modules:      ; $(VERIFY) python scripts/check_module_tests.py
tokens:       ; $(VERIFY) python scripts/check_file_tokens.py
test:         ; $(VERIFY) pytest
preflight:    ; $(VERIFY) python scripts/check_checkout.py && $(VERIFY) python scripts/check_hooks.py
tracked-files: ; $(VERIFY) python scripts/check_tracked_files.py
artifact-sizes: ; $(VERIFY) python scripts/check_artifact_sizes.py
check:        preflight tracked-files artifact-sizes format-check lint typecheck boundaries modules tokens test   # T13 appends: ratchet

# ML targets. They start working with the ticket in brackets.
smoke:        ; $(VERIFY) r0 train --config configs/ci/smoke.toml --out /tmp/r0-smoke              # [T7]
eval-smoke:   ; $(VERIFY) r0 eval --suite dev --checkpoint /tmp/r0-smoke/last.pt --out /tmp/r0-eval  # [T12]
ratchet:      ; $(VERIFY) python scripts/check_ratchet.py                                          # [T13]
mutation:     ; $(VERIFY) python scripts/mutate_diff.py                                            # optional, P2

# Network and real data. Only agent-train.yml (or you, locally) may run these. ADR 0004.
data:         ; uv run r0 data --suite eval/suite_v0.toml                                       # [T9]
train:        ; uv run r0 train --config $(CONFIG) --out checkpoints/$(RUN)                     # [T7]
eval:         ; uv run r0 eval --suite eval/suite_v0.toml --checkpoint checkpoints/$(RUN)/last.pt --out results/runs/$(RUN).json  # [T12]
