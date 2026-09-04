"""``da3 image``/``images``/``video``/``colmap``/``auto`` as real subprocesses.

Every other test in this suite drives the package through its Python API
(`DepthAnything3(...)`, `InferenceService`, direct function calls) or through
typer's `--help` (which only proves the options parse, not that a command
runs). Neither exercises the layer users actually hit: argv parsing, command
routing, reading real files from a real path, and writing real output to a
real directory in a real subprocess. `fix(cli)` in the project history --
`ref_view_strategy` was an undeclared name that broke `image`, `images`,
`colmap` and `video` entirely -- is exactly the kind of regression that
in-process Python-API tests cannot catch and `--help` does not run far
enough to catch either.

Runs against `tiny_checkpoint_dir` (a real `da3-small` saved locally via
`save_pretrained`), `--device cpu`, at a tiny `process_res`. Each test is a
real subprocess launch + real model load, so this file is slower than the
rest of the suite; kept to one smoke test per subcommand rather than a full
option matrix (the option-parsing surface is already covered by `--help` in
CI).
"""

import subprocess
import sys
from pathlib import Path
import imageio
import numpy as np
import pytest
from PIL import Image

CLI = [sys.executable, "-m", "depth_anything_3.cli"]
TIMEOUT = 120


def run_cli(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [*CLI, *args],
        capture_output=True,
        text=True,
        timeout=TIMEOUT,
    )


def _make_image(path: Path, size: int = 64, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    base = rng.integers(0, 255, size=(4, 4, 3), dtype=np.uint8)
    arr = np.kron(base, np.ones((size // 4, size // 4, 1), dtype=np.uint8))
    Image.fromarray(arr).save(path)


@pytest.fixture(scope="module")
def two_images(tmp_path_factory) -> list[Path]:
    directory = tmp_path_factory.mktemp("cli_images")
    paths = []
    for i in range(2):
        path = directory / f"view_{i}.png"
        _make_image(path, seed=i)
        paths.append(path)
    return paths


# ---------------------------------------------------------------------------
# image / images
# ---------------------------------------------------------------------------
def test_image_subcommand_runs_end_to_end(tiny_checkpoint_dir, two_images, tmp_path):
    export_dir = tmp_path / "out"
    result = run_cli(
        "image",
        str(two_images[0]),
        "--model-dir",
        tiny_checkpoint_dir,
        "--device",
        "cpu",
        "--export-dir",
        str(export_dir),
        "--export-format",
        "glb",
        "--process-res",
        "56",
    )
    assert result.returncode == 0, result.stderr
    assert (export_dir / "scene.glb").exists()


def test_images_subcommand_runs_end_to_end(tiny_checkpoint_dir, two_images, tmp_path):
    export_dir = tmp_path / "out"
    result = run_cli(
        "images",
        str(two_images[0].parent),
        "--model-dir",
        tiny_checkpoint_dir,
        "--device",
        "cpu",
        "--export-dir",
        str(export_dir),
        "--export-format",
        "mini_npz",
        "--process-res",
        "56",
    )
    assert result.returncode == 0, result.stderr
    assert any(export_dir.iterdir())


# ---------------------------------------------------------------------------
# video
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def tiny_video(tmp_path_factory) -> Path:
    directory = tmp_path_factory.mktemp("cli_video")
    path = directory / "clip.mp4"
    rng = np.random.default_rng(0)
    with imageio.get_writer(str(path), fps=10, codec="libx264", pixelformat="yuv420p") as writer:
        for _ in range(10):
            base = rng.integers(0, 255, size=(4, 4, 3), dtype=np.uint8)
            frame = np.kron(base, np.ones((16, 16, 1), dtype=np.uint8))
            writer.append_data(frame)
    return path


def test_video_subcommand_runs_end_to_end(tiny_checkpoint_dir, tiny_video, tmp_path):
    export_dir = tmp_path / "out"
    result = run_cli(
        "video",
        str(tiny_video),
        "--model-dir",
        tiny_checkpoint_dir,
        "--device",
        "cpu",
        "--export-dir",
        str(export_dir),
        "--export-format",
        "mini_npz",
        "--process-res",
        "56",
        "--fps",
        "5",
    )
    assert result.returncode == 0, result.stderr
    assert any(export_dir.iterdir())


# ---------------------------------------------------------------------------
# colmap
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def colmap_dir(tmp_path_factory) -> Path:
    """A minimal, hand-written COLMAP text reconstruction: one PINHOLE camera,
    two images at different translations, zero 3D points (the handler only
    needs the per-image pose, not an actual point cloud)."""
    root = tmp_path_factory.mktemp("cli_colmap")
    images_dir = root / "images"
    sparse_dir = root / "sparse"
    images_dir.mkdir()
    sparse_dir.mkdir()

    for i in range(2):
        _make_image(images_dir / f"view_{i}.png", seed=i)

    (sparse_dir / "cameras.txt").write_text("1 PINHOLE 64 64 50.0 50.0 32.0 32.0\n")
    (sparse_dir / "images.txt").write_text(
        "1 1.0 0.0 0.0 0.0 0.0 0.0 0.0 1 view_0.png\n\n"
        "2 1.0 0.0 0.0 0.0 0.0 0.0 -0.5 1 view_1.png\n\n"
    )
    (sparse_dir / "points3D.txt").write_text("")
    return root


def test_colmap_subcommand_runs_end_to_end(tiny_checkpoint_dir, colmap_dir, tmp_path):
    export_dir = tmp_path / "out"
    result = run_cli(
        "colmap",
        str(colmap_dir),
        "--model-dir",
        tiny_checkpoint_dir,
        "--device",
        "cpu",
        "--export-dir",
        str(export_dir),
        "--export-format",
        "mini_npz",
        "--process-res",
        "56",
    )
    assert result.returncode == 0, result.stderr
    assert any(export_dir.iterdir())


# ---------------------------------------------------------------------------
# auto (input-type detection + routing)
# ---------------------------------------------------------------------------
def test_auto_subcommand_routes_a_single_image_like_the_image_command(
    tiny_checkpoint_dir, two_images, tmp_path
):
    export_dir = tmp_path / "out"
    result = run_cli(
        "auto",
        str(two_images[0]),
        "--model-dir",
        tiny_checkpoint_dir,
        "--device",
        "cpu",
        "--export-dir",
        str(export_dir),
        "--export-format",
        "mini_npz",
        "--process-res",
        "56",
    )
    assert result.returncode == 0, result.stderr
    assert any(export_dir.iterdir())


# ---------------------------------------------------------------------------
# error paths: argv/routing failures should exit non-zero, not hang or crash
# with a traceback the way an unguarded NameError would.
# ---------------------------------------------------------------------------
def test_image_subcommand_fails_cleanly_on_a_missing_path(tiny_checkpoint_dir, tmp_path):
    result = run_cli(
        "image",
        str(tmp_path / "does_not_exist.png"),
        "--model-dir",
        tiny_checkpoint_dir,
        "--device",
        "cpu",
        "--export-dir",
        str(tmp_path / "out"),
    )
    assert result.returncode != 0
    assert "not found" in (result.stdout + result.stderr).lower()


def test_images_subcommand_fails_cleanly_on_an_empty_directory(tiny_checkpoint_dir, tmp_path):
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    result = run_cli(
        "images",
        str(empty_dir),
        "--model-dir",
        tiny_checkpoint_dir,
        "--device",
        "cpu",
        "--export-dir",
        str(tmp_path / "out"),
    )
    assert result.returncode != 0
    assert "no image files found" in (result.stdout + result.stderr).lower()
