# Deep modules
- Other code may import only a module's entry point: `r0.modules.<name>` (its `__init__.py`).
- `lib/` is private to its module. Nothing outside the module imports it.
- Tests go through the entry point too: `tests/<name>/` tests `r0.modules.<name>`.
- Inject randomness (`np.random.Generator`) and I/O as arguments; never use globals.
- One module per domain word in CONTEXT.md.
