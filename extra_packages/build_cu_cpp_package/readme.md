This error is very specific:

> `ModuleNotFoundError: No module named 'torch'`

It’s happening inside pip’s **isolated build environment** (the temporary `/tmp/pip-build-env-...` overlay), not inside your `upjab-env`. Even if `torch` is installed in your env, pip **does not use it** for building when `pyproject.toml` exists — it creates a clean build env and installs only what it thinks it needs.

So we need to make sure **torch gets installed into the build env**.

There are two reliable ways.

---

## Option A (recommended): put `torch` in `pyproject.toml` build-system.requires (and ensure pip can install it)

Use this `pyproject.toml`:

```toml
[build-system]
requires = [
    "setuptools>=61.0",
    "wheel",
    "torch"
]
build-backend = "setuptools.build_meta"
```

Then run:

```bash
pip install -U pip setuptools wheel
pip install -e ./extra_packages/build_cu_cpp_package
```

### If it STILL fails with “No module named torch”

That means pip’s build env **couldn’t install torch** (often because your index settings don’t provide it, or you’re on a special platform).

Check what pip is using:

```bash
pip config list
```

If you’re using a custom index that doesn’t host PyTorch wheels, pip can’t fetch torch during build isolation.

In that case, use Option B.

---

## Option B (most robust in real-world dev): disable build isolation for editable install

This forces pip to use your current environment (where torch already exists):

```bash
pip install -e ./extra_packages/build_cu_cpp_package --no-build-isolation
```

This is the common fix for CUDA extensions in research/dev repos.

---

## Best practice for CUDA extensions in editable dev

Do **both** of these:

1. Keep `torch` in `pyproject.toml` (correct metadata)
2. Use `--no-build-isolation` in environments where torch is not pip-installable from your configured index

---

## Quick sanity checks (run these)

### 1) Confirm torch is installed in your active env

```bash
python -c "import torch; print(torch.__version__)"
```

### 2) Confirm pip sees the same python

```bash
which python
which pip
python -c "import sys; print(sys.executable)"
```

If `pip` points to a different environment than `python`, installs/builds go sideways.

---

## Why pip fails *before* building

During “Getting requirements to build editable”, setuptools executes your `setup.py`.
Your `setup.py` does:

```py
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
```

So if torch isn’t available in the **build env**, it fails immediately.

---

## If you want to avoid requiring torch at “requirements discovery”

You can delay importing torch until later (so pip can at least parse requirements), but for PyTorch extensions it’s usually not worth the complexity. The clean approach is Option A/B above.

---

### What I’d do in your situation

Since you already have torch in `upjab-env`:

```bash
pip install -e ./extra_packages/build_cu_cpp_package --no-build-isolation
```

If you paste your `pip config list` output (especially `global.index-url` / `extra-index-url`), I can tell you whether Option A should work without `--no-build-isolation`.
