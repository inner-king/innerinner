# Branch map

This map lists the branches on `origin` as they were on 2026-11-02, after `git fetch origin --prune`. At that time `origin/integration` was `1ef3d71` and `origin/main` was `6ec85ce`. The remote had 55 branches and 27 `archive/*` tags.

- *Commits not in integration* is `git rev-list --count origin/integration..origin/<branch>`: the commits a branch has and `integration` does not.
- A branch with 0 is contained in `integration`: its tip is an ancestor of it, so nothing on the branch is missing there.
- *What it holds* is a short description, or the tip commit's subject where that is the clearest description.

## How to refresh the map

Run these from a clone or worktree of the repository. They only read from `origin`.

```bash
git fetch origin --prune
git ls-remote --heads origin | sed 's#.*refs/heads/##' | while read -r b; do
  n=$(git rev-list --count "origin/integration..origin/$b")
  if git merge-base --is-ancestor "origin/$b" origin/integration; then s=in-integration; else s=not-in-integration; fi
  printf '%s\t%s\t%s\t%s\n' "$b" "$n" "$s" "$(git log -1 --format='%ad %s' --date=short "origin/$b")"
done
git ls-remote --tags origin | sed 's#.*refs/tags/##' | grep -v '\^{}' | grep -c '^archive/'
```

The loop prints one line per branch: name, commits not in integration, whether the tip is contained in `integration`, and the tip's date and subject. The last line counts the archive tags. Restore one with the command in [CONTRIBUTING.md](../CONTRIBUTING.md#branches).

## Base branches

| Branch | Commits not in integration | What it holds |
| --- | ---: | --- |
| `integration` | 0 | Default branch on GitHub and the base for new work. |
| `main` | 0 | Frozen at the pipeline baseline (6ec85ce); integration already contains it. |

## Topic branches not in integration yet (open work)

These codelead topics are on the remote and `integration` does not contain them. This file does not record their review status.

| Branch | Commits not in integration | What it holds |
| --- | ---: | --- |
| `codelead/bg-align-appearance` | 2 | Background-splat appearance alignment to the scene (WIP), with mutation-test gaps closed. |
| `codelead/dataset-root` | 1 | An explicit dataset root for the pipeline, and a record of where stage 13 wrote its assets. |
| `codelead/stage6-seam` | 1 | Stage 6: the composite seam only where the infill filled something in. |

Local only, not on the remote when this map was made: `codelead/solver-r3` (an opt-in subgrid blade-face contact for the cutting solver) and `codelead/solver-r5` (vegetable material presets and run warnings). Their content reached `integration` as one merge (`59f95fc`) with the push that carries this file, so the two branches themselves stay local. `codelead/solver-r4` (validation scenes and the plate hold) is still in progress and local only.

## Topic branches already in integration (merged; safe to ignore)

| Branch | Commits not in integration | What it holds |
| --- | ---: | --- |
| `codelead/asset-naming` | 0 | Stages 11 and 13: name each asset by its input, and say when an import would replace an asset another scene uses |
| `codelead/bg-attach-stopped` | 0 | README: stage 14's stopped-block attach is confirmed by a real run |
| `codelead/bg-exposure` | 0 | Stage 14: optional exposure for the background mesh (lit multiplier or unlit emission) |
| `codelead/bg-mesh-comment` | 0 | Stage 5c: say the 25 cracked pieces at a 2 cm snap were not reproduced |
| `codelead/bg-mesh-quality` | 0 | Stage 5c: fill the pixels under each object from the surface it stood on, and drop border and tiny pieces |
| `codelead/bg-mesh-scene-attach` | 0 | Background USD: hang the mesh off a rigid-body link, which OmniGibson requires |
| `codelead/cpd-scale-limit` | 0 | Stage 8: an optional limit on the scale CPD adds to the pre-scaled mesh, off by default |
| `codelead/depth-mesh-background` | 0 | Add an optional stage 5c: a textured background mesh from the canonical frame's depth |
| `codelead/fit-axis` | 0 | Stage 8: three optional safeguards so the height fit does not flatten thin round objects |
| `codelead/hydra-module-loggers` | 0 | Hydra: stop silencing the stage modules' INFO and WARNING lines |
| `codelead/image-backend-name` | 0 | Local VLM: name the qwen21 image model after the UNET file it loads |
| `codelead/merge-preview-batch-b` | 0 | merge codelead/plane-snap-ramp (B18c: README keeps both sides, the tapered move stays with the one-sided snap section) |
| `codelead/ollama-batch-pin` | 0 | Local VLM: let a run pin Ollama's num_batch and num_gpu, key the call cache on them, and record them |
| `codelead/plane-determinism` | 0 | Stages 3 and 5: make the plane fits and the removal order's random draws repeat for a seed |
| `codelead/plane-snap` | 0 | Stage 5c: let the plane snap reach further under the support plane than over it |
| `codelead/plane-snap-ramp` | 0 | Stage 5c: fade a one-sided plane snap in from the rim of the snapped region |
| `codelead/pre-scale-method` | 0 | Stage 8: an optional minimal-volume box for the pre-scale factor, off by default |
| `codelead/stage-info-models-3-9-11` | 0 | Stages 3, 9 and 11: record the model that answered in stage_info.json |
| `codelead/stage11-mass-flags` | 0 | Stage 11: flag a mass that was held at the clamp edge or that no material of its family could have |
| `codelead/stage6-infill-color` | 0 | Stage 6: two options against the upsampler's colour drift, off by default |
| `codelead/stage6-reference-check` | 0 | Stage 6: record how far each upsampled silhouette is from the stage-5 crop it came from |
| `codelead/stage8-reproducible` | 0 | Stage 8: seed every random source from the config seed, once per object |
| `codelead/stage8-size-followup` | 0 | Stage 8: record the scale the object was placed at, and finish the size-check tests |
| `codelead/stage8-size-warning` | 0 | Stage 8: compare each placed mesh's footprint and height with its depth, and warn |

## Research lines

| Branch | Commits not in integration | What it holds |
| --- | ---: | --- |
| `research1/b10-check` | 1 | WIP stage 8 size check: stretch node-transformed GLBs, tolerate floor noise, read a fallback floor. |
| `research1/b6test` | 0 | Background USD: hang the mesh off a rigid-body link, which OmniGibson requires |
| `research1/b7test` | 1 | Stage 5c: fill the pixels under each object from the surface it stood on; drop border and tiny pieces. |
| `research1/hfit-b2b` | 1 | WIP stage 8 size check, with the same subject line as research1/b10-check (a different commit). |
| `research1/lidar-colfit` | 1 | WIP stage 8: an optional column fit that lays a slab-like mesh between its support and the depth's top. |
| `research1/mergepreview-batch-b` | 0 | merge codelead/stage11-mass-flags |
| `research1/yt-veg` | 0 | merge codelead/plane-snap-ramp (B18c: README keeps both sides, the tapered move stays with the one-sided snap section) |
| `research2/b2-check` | 0 | Stage 8: compare each placed mesh's footprint and height with its depth, and warn |
| `research2/b8-run` | 0 | Stage 6: two options against the upsampler's colour drift, off by default |
| `research2/preview-full` | 0 | merge codelead/pre-scale-method (README: the stage 8 paragraph before the stage 3 and 5 reproducibility section) |
| `research3/tofu-policy` | 0 | Tofu cutting: add the episode runner for demonstrations and policy evaluation |

## Older `claude/*` session branches

Cutting history and early reproduction work. Those with 0 commits are contained in `integration`.

| Branch | Commits not in integration | What it holds |
| --- | ---: | --- |
| `claude/beautiful-saha-73ae5c` | 7 | B7 replays (30 seeds) on the Marker-in-Cup scenes, and the per-pipeline VLM defaults. |
| `claude/cool-taussig-978a2f` | 0 | Tofu cutting: a Franka in the Chef'd OmniGibson scene cuts the MPM tofu |
| `claude/dazzling-poincare-109cdf` | 6 | B7 cross-input and replays on the Marker-in-Cup scenes, and the per-pipeline VLM defaults. |
| `claude/eager-meitner-113801` | 1 | pi0.5 on Marker in Cup: start distribution and action execution test. |
| `claude/elastic-bouman-b393f6` | 2 | pi0.5 on Marker in Cup: the tight-grid, horizon-15 lift at 100 episodes. |
| `claude/eloquent-mcclintock-54a2a7` | 0 | Runtime cutting with the knife in the Franka's hand: 19 of 19 strokes cut |
| `claude/focused-hellman-26d6d9` | 0 | Fruit slicing: measure runtime cutting costs on Isaac Sim 5.1 |
| `claude/hanbin5-local-comfyui` | 4 | ComfyUI reinstall record and local setup notes for the local image backend (the same commits as hanbin5/local). |
| `claude/magical-tereshkova-c02a63` | 0 | Tofu cutting: grow the crack by an energy balance, consistent across grids |
| `claude/modest-ptolemy-98b0ab` | 2 | A chapter 19.4 note fix, and the DreamZero client matched to DreamZero's public server. |
| `claude/musing-dijkstra-fe4b88` | 0 | Tofu cutting: let a partly cut block tear |
| `claude/object-cutting-simulation-research-23a87c` | 0 | Tofu cutting: a Franka in the Chef'd OmniGibson scene cuts the MPM tofu |
| `claude/optimistic-shamir-ad0823` | 7 | B7 replays (30 seeds) on the Marker-in-Cup scenes, and the per-pipeline VLM defaults (same tip as claude/beautiful-saha-73ae5c). |
| `claude/pensive-clarke-79d49e` | 0 | README 15.2: numbers from a run of the committed code |

## `hanbin5/*`

| Branch | Commits not in integration | What it holds |
| --- | ---: | --- |
| `hanbin5/local` | 4 | ComfyUI reinstall record and local setup notes for the local image backend (the same commits as claude/hanbin5-local-comfyui). |
