# Contributing to SimFoundry

This page says which branch to use, where things live, how to run the tests and how to send a change. The rules for working in this tree, such as not editing `deps/` and the saved-scene format, are in [AGENTS.md](AGENTS.md).

## Branches

- `integration` is the default branch on GitHub and the base for new work. Branch from it and open pull requests against it.
- `main` is frozen at the pipeline baseline, commit `6ec85ce`. It is not the place for new work.
- Topic branches are named `<area>/<topic>`:
  - `codelead/<topic>` for single-topic changes that were reviewed before they were merged;
  - `research1/<topic>`, `research2/<topic>` and `research3/<topic>` for research work.
- Tags `archive/<name>` keep the commits of branches that were deleted because `main` already had their content. To restore one, run `git branch <name> archive/<name>`, for example `git branch codelead/hunyuan-seed archive/codelead/hunyuan-seed`.

[docs/BRANCHES.md](docs/BRANCHES.md) maps the branches on the remote, says which ones `integration` does not contain yet, and gives the commands that refresh the map.

## Where things are

- Pipeline stages: [scripts/pipeline/A_reconstruction/stages/](scripts/pipeline/A_reconstruction/stages/) (reconstruction), [scripts/pipeline/B_augmentation/stages/](scripts/pipeline/B_augmentation/stages/) (augmentation) and [scripts/pipeline/C_application/stages/](scripts/pipeline/C_application/stages/) (application).
- MPM cutting solver for soft food: [simfoundry/cutting/mpm.py](simfoundry/cutting/mpm.py), with its material presets in `simfoundry/cutting/materials.py`.
- Runtime (rigid) cutting: [simfoundry/cutting/rule.py](simfoundry/cutting/rule.py), a knife that cuts where it is, as a transition rule that replaces an object by two pieces. The food-knife co-simulation with OmniGibson is in `simfoundry/cutting/cosim.py` and `simfoundry/cutting/og_cosim.py`.
- Policy tooling: [scripts/cutting/policy/](scripts/cutting/policy/), whose README lists the scripts.
- Scene editors, both described in [docs/INSTRUCTIONS_SCENE_EDITOR.md](docs/INSTRUCTIONS_SCENE_EDITOR.md): the browser editor in [scripts/interactive/light_editor/](scripts/interactive/light_editor/), and the OmniGibson editor, [scripts/interactive/interactive_scene_editor.py](scripts/interactive/interactive_scene_editor.py).
- [AGENTS.md](AGENTS.md) is the rules page: do not edit `deps/` (OmniGibson is not tracked by this repository), the saved-scene contract, and the coordinate conventions.

## Running the tests

Run the tests from the root of the checkout you are changing, in a Python environment that has the repository's dependencies. The suite runs on the CPU, so hide the GPUs with `CUDA_VISIBLE_DEVICES=`.

```bash
export CUDA_VISIBLE_DEVICES=     # CPU only
export ENFORCE_FAISS_GPU=0       # let FAISS search fall back to the CPU (by default it requires a GPU)
export PYTHONPATH="$PWD"         # import this checkout's simfoundry package
python -B -m pytest -p no:cacheprovider -rfEsx                              # the whole suite; -rfEsx lists failures, errors and skips
python -B -m pytest -p no:cacheprovider tests/test_served_model_names.py -q # one file
```

Before you trust a result, check which copy of `simfoundry` Python imports. From the checkout root:

```bash
python -B -c 'import simfoundry; print(simfoundry.__file__)'
```

The path it prints must be inside your checkout. If it points at another clone, fix `PYTHONPATH` or the environment first. `pytest.ini` also puts the repository root on the import path for pytest. Scripts run as files (`python scripts/....py`) do not get it, so keep `PYTHONPATH` set for them.

Some tests need a package that is not installed. They skip through `pytest.importorskip`, and `-rs` (included in `-rfEsx`) gives the reason. The optional packages are `pxr` (OpenUSD), `trimesh`, `open3d`, `torch`, `warp` (NVIDIA Warp, which the cutting solver uses), `cv2`, `scipy`, `skimage`, `yaml`, `omegaconf`, `shapely` and `hydra`.

## Sending a change

- Open a pull request against `integration`. Keep one topic in each pull request, on its own `<area>/<topic>` branch.
- Add tests for new behaviour, and check that a new test fails before the fix.
- Put new behaviour behind an option that is off by default. Where the change can keep the default output byte-identical, show it: run the default before and after and compare the outputs.
- Do not add files over 1 MB. Do not add PNG or GIF files under `docs/`: `.gitattributes` stores them with Git LFS. MP4 files are committed directly.
- Do not commit keys, tokens, personal paths or machine names.
- Sign off every commit with `git commit -s`, as the DCO section below requires. In a clone, install the whitespace hook once from the repository root with `ln -sf ../../scripts/hooks/pre-commit .git/hooks/pre-commit`. It refuses trailing whitespace, a blank line at the end of a file and conflict markers.

#### Signing Off Your Work

**Note**: This project will only accept contributions under Apache-2.0.

* We require that all contributors "sign-off" on their commits. This certifies that the contribution is your original work, or you have rights to submit it under the same license, or a compatible license.

  * Any contribution which contains commits that are not Signed-Off will not be accepted.

* To sign off on a commit you simply use the `--signoff` (or `-s`) option when committing your changes:
  ```bash
  $ git commit -s -m "Add cool feature."
  ```
  This will append the following to your commit message:
  ```
  Signed-off-by: Your Name <your@email.com>
  ```

* Full text of the DCO (https://developercertificate.org/):

  ```
    Developer Certificate of Origin
    Version 1.1

    Copyright (C) 2004, 2006 The Linux Foundation and its contributors.

    Everyone is permitted to copy and distribute verbatim copies of this
    license document, but changing it is not allowed.


    Developer's Certificate of Origin 1.1

    By making a contribution to this project, I certify that:

    (a) The contribution was created in whole or in part by me and I
        have the right to submit it under the open source license
        indicated in the file; or

    (b) The contribution is based upon previous work that, to the best
        of my knowledge, is covered under an appropriate open source
        license and I have the right under that license to submit that
        work with modifications, whether created in whole or in part
        by me, under the same open source license (unless I am
        permitted to submit under a different license), as indicated
        in the file; or

    (c) The contribution was provided directly to me by some other
        person who certified (a), (b) or (c) and I have not modified
        it.

    (d) I understand and agree that this project and the contribution
        are public and that a record of the contribution (including all
        personal information I submit with it, including my sign-off) is
        maintained indefinitely and may be redistributed consistent with
        this project or the open source license(s) involved.
  ```