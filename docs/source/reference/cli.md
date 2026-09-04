# Command Line Interface

Every option below is rendered directly from the `da3` CLI's own Typer/Click
definitions, so it can't drift out of sync with what `--help` prints.
Task-oriented example commands follow in {ref}`examples-cli` below.

```{eval-rst}
.. typer:: depth_anything_3.cli:app
   :prog: da3
   :show-nested:
   :width: 100
```

(examples-cli)=

## Examples

The CLI supports image, image-directory, video, and COLMAP-dataset
processing, plus a Gradio web UI and gallery viewer. A backend service can
keep the model resident in GPU memory across jobs, so it doesn't reload for
every command.

### Quick start

```bash
# Start backend service (optional, keeps model resident in GPU memory)
da3 backend --model-dir depth-anything/DA3NESTED-GIANT-LARGE-1.1

# Auto mode: detect input type and process (no backend needed)
da3 auto path/to/input --export-dir ./workspace/scene001

# Reuse the backend for the next job
da3 auto path/to/video.mp4 \
    --export-dir workspace/gallery/scene002 \
    --use-backend \
    --backend-url http://localhost:8008
```

Each export directory contains `scene.glb`, `scene.jpg`, and optional
extras such as `depth_vis/` or `gs_video/` depending on the requested
format.

```{note}
When a job is submitted to a running `da3 backend` over HTTP,
`--export-dir` should be a relative path under that backend's
`--gallery-dir` (`workspace/gallery` by default). This doesn't apply to
local (non-backend) runs, where `--export-dir` can be anywhere writable.
```

### `auto` — detect and dispatch

`da3 auto INPUT_PATH [OPTIONS]` inspects `INPUT_PATH` and routes to the
matching handler:

- a single image file (`.jpg`, `.png`, `.jpeg`, `.webp`, `.bmp`, `.tiff`,
  `.tif`)
- an image directory
- a video file (`.mp4`, `.avi`, `.mov`, `.mkv`, `.flv`, `.wmv`, `.webm`,
  `.m4v`)
- a COLMAP directory (containing `images/` and `sparse/`)

```bash
da3 auto path/to/image.jpg --export-dir ./output
da3 auto path/to/video.mp4 --fps 2.0 --export-dir ./output
da3 auto path/to/input --export-format mini_npz-glb --use-backend --export-dir ./output
```

### `image` / `images` / `video` / `colmap`

Process one image, a directory of images, a video (frames are extracted
first), or a COLMAP reconstruction directly:

```bash
da3 image path/to/image.png --export-dir ./output
da3 images ./image_folder --export-dir ./output
da3 video path/to/video.mp4 --fps 2.0 --process-res 1024 --export-dir ./output
da3 colmap ./colmap_dataset --sparse-subdir 0 --export-dir ./output
```

Feature visualization and Gaussian Splatting work the same way as the
Python API -- combine export formats with `-`:

```bash
da3 image image.jpg --export-format feat_vis --export-feat "9,19,29,39" --export-dir ./results
da3 auto video.mp4 --export-format glb-feat_vis --export-feat "11,21,31" --export-dir ./debug --use-backend
```

### `backend` — keep the model resident

```bash
da3 backend --model-dir depth-anything/DA3NESTED-GIANT-LARGE-1.1
```

Keeps the model in GPU memory, exposes a REST inference API, and (with
`--gallery-dir`) a gallery browser. Every export from a backend-served
request lands under the configured `--gallery-dir`.

**Authentication.** By default the server binds `127.0.0.1` (localhost-only)
with no authentication. Binding `--host` to anything else without
`--api-key` makes it generate and print a random key on startup:

```text
No --api-key was set, so one was generated for this run:
  X-API-Key: FdGcurw5z45kuGBjd5wO5S1agAf-iknYnfHhpWA3RmA
```

Copy that into `DA3_BACKEND_API_KEY` for anything that submits jobs to it.
For a key that survives restarts:

```bash
export DA3_BACKEND_API_KEY=$(openssl rand -hex 32)
da3 backend --host 0.0.0.0 --model-dir depth-anything/DA3NESTED-GIANT-LARGE-1.1
```

`da3 ... --use-backend` reads `DA3_BACKEND_API_KEY` from its own
environment automatically. Pass `--allow-unauthenticated` to skip
authentication on a network you already trust.

### `gradio` and `gallery`

```bash
da3 gradio --model-dir depth-anything/DA3NESTED-GIANT-LARGE-1.1 \
    --workspace-dir ./workspace --gallery-dir ./gallery

da3 gallery --gallery-dir ./workspace --open-browser
```

The gallery expects each scene folder to contain at least `scene.glb` and
`scene.jpg`, with optional subfolders like `depth_vis/` or `gs_video/`.

### Workflow examples

**Batch processing with a shared backend:**

```bash
da3 backend --model-dir depth-anything/DA3NESTED-GIANT-LARGE-1.1 \
    --host 0.0.0.0 --port 8008 --gallery-dir ./workspace

for scene in scene1 scene2 scene3; do
    da3 auto ./data/$scene --export-dir ./workspace/$scene --use-backend --auto-cleanup
done

da3 gallery --gallery-dir ./workspace --open-browser
```

**Multiple export formats plus custom resolution:**

```bash
da3 image image.jpg \
    --process-res 1024 \
    --num-max-points 2000000 \
    --conf-thresh-percentile 30.0 \
    --export-format mini_npz-glb \
    --export-dir ./output
```

### Getting help

```bash
da3 --help
da3 auto --help
da3 image --help
```
