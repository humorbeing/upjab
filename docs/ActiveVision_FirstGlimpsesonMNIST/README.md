# MNIST Patch Capture

Static GitHub Pages app for exploring MNIST patch capture. It shows the original digit, a square crop, the crop resized to `N x N`, the resized patch reconstructed into the original crop area, and an accumulated remembered view.

## Instructions

Move the cursor over Panel 1, Panel 4, or Panel 5 to move the crop area. Use Up and Down to change crop size, and Left and Right to change N size. Left-click to remember the current glimpse in Panel 5; right-click to reset.

Black out Panels 1-4 to hide different parts of the process. The challenge is to use the fewest accumulated pixels while still reconstructing a recognizable digit in Panel 5.

## Files

- `index.html`, `style.css`, `app.js`: static browser app
- `samples/mnist_samples.json`: 100 bundled MNIST samples, 10 per digit
- `make_samples.py`: optional asset preparation script

## Generate Samples

```bash
python3 make_samples.py
```

The script prefers `torchvision.datasets.MNIST(root="./data", train=True, download=True)`. If `torchvision` is unavailable but raw MNIST IDX files already exist at `data/MNIST/raw`, it can generate the same JSON from those local files.

## Run Locally

Because browsers usually block `fetch()` from local `file://` pages, serve the folder with a tiny static server:

```bash
python3 -m http.server 8000
```

Then open `http://localhost:8000/`.

## Deploy To GitHub Pages

Commit the project files and enable GitHub Pages for the branch or folder that contains `index.html`. No backend, Python runtime, Node server, or build step is required at deployment time.
