## Transfer Bundle

This folder is for cross-machine handoff only.

Included:

- `master_pack.zip`: zipped local `master_pack/master_pack` dataset bundle
- `dada_annotations.xlsx`: local DADA annotation spreadsheet copy

Notes:

- `master_pack.zip` is tracked by Git LFS.
- After cloning on AutoDL or another machine, run:

```bash
git lfs pull
```

- Then extract `master_pack.zip` to the target data directory before running experiments.
