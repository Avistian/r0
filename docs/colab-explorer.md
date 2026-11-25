# Explorer inside a hosted Colab notebook

[The generation notebook](../notebooks/generation.ipynb) installs the pinned adapter
revision `8794ad9581eafa3fa17ef77d279613058512acdc`. It offers all ten starting recipes
for bounded generation, measurement and larger resumable runs, followed by the
embedded explorer. The explorer uses the same controls, charts and session actions
as the local application. Its CPU/CUDA selector is independent of the bulk-run cell.

```python
from pathlib import Path
from google.colab import output
from r0.modules.explorer import show_colab_explorer

callback = show_colab_explorer(Path("/content/drive/MyDrive/r0-generation"))
# When replacing the output: output.unregister_callback(callback)
```

The adapter registers a session callback and displays the shared HTML assets in the
executed Colab output. Browser actions call that callback; no local HTTP server is
required. Colab/IPython are supplied by the hosted runtime and are imported only
when the notebook adapter is called. Replacing the output should unregister its
old callback so it releases its dataset. The notebook does this automatically on
cell rerun. After a new runtime starts, execute the setup and explorer cells again.

The transport follows Google's [advanced outputs example](https://colab.research.google.com/notebooks/snippets/advanced_outputs.ipynb)
and public [callback API](https://github.com/googlecolab/colabtools/blob/main/google/colab/output/_js.py).
Callbacks are available to executed, trusted output cells in the current session.

Local tests substitute only the external Colab/IPython APIs. They verify callback
registration, shared markup, structured results and visible errors. They do not
establish hosted interaction, Drive durability, GPU operation or download behavior.
The notebook contains a hosted verification checklist. Those checks remain
**NOT_RUN** until performed in an actual Colab runtime. No standalone hosted website
is part of this request; the requested hosted explorer is embedded in Colab.

The smoke cell retains its fresh timing before the estimate cell runs. A resumed
smoke skips completed tasks and uses that retained timing, clearly labelled as an
earlier measurement. If no timing survived, the estimate stops with instructions
to choose a new seed explicitly; it never regenerates completed tasks silently.
The [local notebook walkthrough](evidence/issue-27-colab-explorer/README.md) covers
this cell boundary for linear and composed recipes.
