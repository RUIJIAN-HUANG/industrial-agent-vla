# Test fixtures

`golden_episode_v1/` is the compact, immutable Canonical Episode baseline used
by recorder, reader, replay, and split-registry tests. It contains three
synchronized 1280x720 RGB streams, both frozen 7-D robot-state streams, and one
valid 7-D action chunk. It contains no wrist-camera image and no offline ground
truth.

Generate it once from a clean fixture directory with:

```powershell
python scripts/generate_golden_episode.py
```

The generator refuses to overwrite an existing baseline. Review and commit any
intentional replacement as a new fixture version.
