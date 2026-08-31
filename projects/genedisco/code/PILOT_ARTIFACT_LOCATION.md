# IL-2 pilot artifact location

The original source tree used absolute symlinks into the laboratory network share. They were intentionally replaced in this continuity repository: absolute symlinks would be broken on every new machine and would make the migration depend on `/mnt/Omics_GPU`.

The selected 24-job IL-2 three-seed logs and results are included in the private Drive artifact:

```text
genedisco-source-and-il2-pilot-20260831.tar.zst
```

Look up its Drive path and SHA-256 in the root `manifests/artifacts.yaml`, verify it, then extract it into the chosen restoration directory. The public GeneDisco dataset itself remains redownloadable through the version/hash instructions in `REPRODUCTION.md`.
