# Project Scope

This repository is a curated portfolio/export repository extracted from the
larger CR3 + CRAFT / DexJoCo workspace.

It is intentionally smaller than the full workspace.

## Included

- Original CR3+CRAFT DexJoCo integration code excerpts.
- Teleoperation scripts written for CR3+CRAFT testing.
- Windows uv environment setup for MediaPipe-based validation.
- The small MediaPipe hand landmark task model used by the included camera path.
- Documentation and handoff reports.
- Small rendered media assets for GitHub presentation.

The repository now includes the standalone `dexjoco/` package, the CR3+CRAFT
MuJoCo assets, controllers, task XMLs, scripts, and Windows setup files. It is
usable without merging files into another DexJoCo checkout.

## Excluded

- Third-party repositories copied into the local workspace.
- The full DexJoCo base package and its unrelated task assets.
- Large model weights.
- Vendor installers.
- CAD source files and 3D-print files.
- Local virtual environments and caches.

## Upload Intent

The purpose is to enrich a GitHub portfolio with a readable project narrative,
selected source code, and visual proof-of-work without accidentally publishing
large or licensed third-party assets.
