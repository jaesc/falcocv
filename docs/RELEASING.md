# Releasing FalcoCV

## 1. Publish the model release

The wheel/sdist intentionally exclude `models/` (~129 MB). The model files must
be available as a **public** GitHub Release before publishing to PyPI: assets in
a private repository require authentication and the downloader is anonymous.

1. Create the public repository `jaesc/falcocv` (or adjust the URL below and in
   `src/falcocv/core/models.py`).
2. Create a release tagged `models-v1`.
3. Upload every file listed in `MODEL_FILES` from `src/falcocv/core/models.py`
   (`EfficientNet-B0*`, `ResNet18-*`, `ch_PP-OCRv5_rec_*`, `ppocrv5_dict.txt`)
   with the exact file names, plus a `NOTICE`/`LICENSE` for the weight
   provenance described in `CREDITS.md`.
4. Verify an unauthenticated download works:

   ```bash
   curl -I https://github.com/jaesc/falcocv/releases/download/models-v1/EfficientNet-B0.onnx
   ```

If you host the models on a mirror, set `FALCOCV_MODELS_URL` for users or change
`DEFAULT_MODELS_URL` in `src/falcocv/core/models.py`.

## 2. Local release checks

```bash
python -m pip install -e ".[dev]"
pytest -q --ignore=tests/test_trainer_equivalence.py
falcocv self-test                                      # installation smoke test
python -m build --sdist --outdir dist
python scripts/check_wheel.py                          # wheel contents
python scripts/wheel_smoke.py                          # install wheel in a temp venv + self-test
```

The sdist must be < 100 MB (currently ~1.1 MB) and contain `cpp/kernel/` +
`src/falcocv/` but no `models/`. The wheel must contain the LICENSE file and the
`License-Expression` metadata (GPL-3.0-or-later).

Build the sdist from a **clean checkout**: PEP 639 license globs bypass
`sdist.exclude`, so a local `python -m build` in a working copy also picks up
nested `LICENSE` files from `.venv`:

```bash
git archive --format=zip HEAD -o /tmp/falcocv-src.zip
unzip -q /tmp/falcocv-src.zip -d /tmp/falcocv-src
cd /tmp/falcocv-src && python -m build --sdist --wheel --outdir dist
```

## 3. Build platform wheels

Push a tag (for example `v1.0.0rc1`) or run the **Wheels** GitHub Actions
workflow manually. It builds:

- Windows x86_64 wheels (CPython 3.13.x)
- Linux x86_64 manylinux_2_34 wheels (CPython 3.13.x; PyQt6 needs glibc 2.34+)
- an sdist

Download the artifacts and inspect that the kernel is inside the package:

```bash
python -c "import zipfile; print([n for n in zipfile.ZipFile('falcocv-...whl').namelist() if '_falco_kernel' in n])"
```

## 4. Trusted publishing (one-time setup)

TestPyPI and PyPI are separate accounts.

1. Create an account at <https://test.pypi.org/account/register/>, verify the
   email and enable two-factor authentication.
2. Add a **pending trusted publisher** at
   <https://test.pypi.org/manage/account/publishing/>:

   | Field | Value |
   |---|---|
   | PyPI Project Name | `falcocv` |
   | Owner | `jaesc` |
   | Repository name | `falcocv` |
   | Workflow name | `publish.yml` |
   | Environment name | `testpypi` |

3. Repeat on <https://pypi.org/account/register/> and
   <https://pypi.org/manage/account/publishing/> with environment `pypi`.
   The automatic tag release uses the **Release** workflow, so add a second
   publisher for the same project with:

   | Field | Value |
   |---|---|
   | PyPI Project Name | `falcocv` |
   | Owner | `jaesc` |
   | Repository name | `falcocv` |
   | Workflow name | `release.yml` |
   | Environment name | `pypi` |

   PyPI allows multiple trusted publishers for one project, so the manual
   `publish.yml` fallback keeps working.

4. Recommended: create the GitHub environments `testpypi` and `pypi`
   (Settings → Environments) and add required reviewers to `pypi`.

No API token is stored anywhere: the `Publish` and `Release` workflows
authenticate with OIDC (`id-token: write`).

## 5. Automatic release from a tag

Pushing a tag that matches `src/falcocv/version.py` (for example `v1.3.0`)
runs **Actions → Release**. It builds, in parallel:

- CPython 3.13 wheels for Windows x64 and Linux x86_64 (manylinux_2_34),
- the sdist,
- the Windows standalone bundle, ZIP and Inno Setup installer,
- the Linux standalone bundle, ZIP and AppImage.

Then it:

1. publishes the wheels and sdist to PyPI;
2. generates `SHA256SUMS.txt`;
3. signs `SHA256SUMS.txt` and the AppImage with the FalcoCV GPG key when both
   GPG secrets exist;
4. creates or updates the GitHub release and uploads every asset.

Windows Authenticode signing is added once SignPath Foundation approves the
project. Until then the Windows binaries are unsigned and the release still
succeeds.

### GPG secrets (one-time setup)

Generate or import the release key and add two repository secrets
(Settings → Secrets and variables → Actions):

| Secret | Value |
|---|---|
| `GPG_PRIVATE_KEY` | ASCII-armored private key (`gpg --armor --export-secret-keys <fingerprint>`) |
| `GPG_PASSPHRASE` | passphrase of that key |

The matching public key is committed as `docs/FALCOCV_GPG_KEY.asc`. Its
fingerprint must match the one in `docs/DOWNLOAD.md`
(`3F5717C512E646A37E22D6B7B2BE6A6DB85C6837`).

If either secret is missing, the release is still published without the GPG
signatures; the signing step is skipped.

## 6. Publish a release candidate

Actions → **Publish** → *Run workflow* → target `testpypi`.

Validate in a clean venv. TestPyPI has no dependencies, and `--extra-index-url`
can pull unrelated pre-releases from TestPyPI, so download the wheel with
`--no-deps` and install it against regular PyPI:

```bash
python -m venv /tmp/falcocv-rc
/tmp/falcocv-rc/bin/pip download --no-deps -i https://test.pypi.org/simple/ \
    -d /tmp/falcocv-wheel falcocv
/tmp/falcocv-rc/bin/pip install /tmp/falcocv-wheel/falcocv-*.whl
/tmp/falcocv-rc/bin/falcocv self-test
```

Check `falcocv` (editor), `falcocv-server`, a real job through
`falcocv run job.cvj --output results.json`, and the first-run model download on
Windows and Linux.

## 7. Publish the stable release

1. Bump `src/falcocv/version.py` only (for example `1.2.0` → `1.3.0`);
   `pyproject.toml` reads the version dynamically from that file.
2. Commit and push `main`.
3. Create and push the tag:

   ```bash
   git tag v1.3.0
   git push origin v1.3.0
   ```

4. **Actions → Release** builds every artifact, publishes to PyPI and creates
   the GitHub release. The tag must match `version.py`, otherwise the workflow
   stops before publishing.

The older manual path remains available as a fallback: Actions → **Publish** →
target `pypi`, then create the GitHub release manually. Manual upload with a
token also works: `twine upload --repository testpypi dist/*` and
`twine upload dist/*`.

## 8. Post-release smoke test

In a clean environment:

```bash
python -m pip install falcocv
falcocv download-models
falcocv self-test
```
