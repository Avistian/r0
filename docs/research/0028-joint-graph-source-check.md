# Graph-integration cached source checksum procedure

Executed 2026-11-27 locally from `/tmp/r0-28-graph-repair` on Linux
6.6.87.2-microsoft-standard-WSL2, GNU coreutils 9.4, Python 3.12.3.
This verifies the two retained local files only, not their current upstream state.

Command:

```sh
sha256sum /tmp/r0-28-source-read/I2func.py /tmp/node-pipeline-sampling-research/_points.py
```

Exact stdout (also retained at `/tmp/r0-28-graph-source-sha256.txt`):

```text
ff2939197b353f6d11a3fbaf0e27586998bdf93527dad11f24698ff296ce2f6c  /tmp/r0-28-source-read/I2func.py
04dcbd7664004f6acc8453ca866caba0879d87d4ecbcdad08dd79f9ca2ff7a04  /tmp/node-pipeline-sampling-research/_points.py
```

Exit status 0. The research note's source attribution remains inherited from earlier
extraction. No fresh upstream retrieval was performed for this check.
