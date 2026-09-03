"""The export dispatcher and the writers behind it.

``utils/export/__init__.py`` is a hand-written if/elif chain sitting next to a
``SUPPORTED_EXPORT_FORMATS`` set that is documented as the server-trusted
enum for request validation. Nothing keeps the two in step, and
``--export-format`` is a free string on the CLI, so drift means either a
format the CLI advertises and silently refuses, or one that bypasses
validation. The first test here compares the set to the branches the
dispatcher actually implements.

The rest exercise real writes to a temporary directory and read the results
back, so the file layout and dtypes an external consumer depends on are
pinned rather than assumed.
"""

import ast
import inspect
import os
import time
from pathlib import Path
import numpy as np
import pytest

from depth_anything_3.specs import Prediction
from depth_anything_3.utils import export as export_mod
from depth_anything_3.utils.export import SUPPORTED_EXPORT_FORMATS, export

N, H, W = 3, 8, 10


def _prediction(**overrides) -> Prediction:
    rng = np.random.default_rng(0)
    extrinsics = np.tile(np.eye(4), (N, 1, 1))
    extrinsics[:, :3, 3] = rng.normal(size=(N, 3))
    intrinsics = np.tile(np.eye(3), (N, 1, 1))
    intrinsics[:, 0, 0] = intrinsics[:, 1, 1] = 6.0
    intrinsics[:, 0, 2], intrinsics[:, 1, 2] = W / 2, H / 2
    fields = dict(
        depth=rng.uniform(0.5, 5.0, size=(N, H, W)).astype(np.float32),
        is_metric=0,
        conf=rng.uniform(1.0, 3.0, size=(N, H, W)).astype(np.float32),
        extrinsics=extrinsics,
        intrinsics=intrinsics,
        processed_images=rng.integers(0, 255, size=(N, H, W, 3), dtype=np.uint8),
    )
    fields.update(overrides)
    return Prediction(**fields)


def _wait_for(path: Path, timeout: float = 60.0) -> Path:
    """The npz writers are decorated with ``@async_call``: they start a thread
    and return immediately, so the caller has to wait for the file."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.exists() and path.stat().st_size > 0:
            time.sleep(0.05)  # let the writer finish flushing
            return path
        time.sleep(0.02)
    raise AssertionError(f"{path} was never written")


# ---------------------------------------------------------------------------
# the dispatch table
# ---------------------------------------------------------------------------
def _implemented_formats() -> set[str]:
    """Every literal compared against ``export_format`` inside ``export``."""
    tree = ast.parse(inspect.getsource(export))
    found = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Compare) and isinstance(node.ops[0], ast.Eq):
            left, right = node.left, node.comparators[0]
            if isinstance(left, ast.Name) and left.id == "export_format":
                if isinstance(right, ast.Constant) and isinstance(right.value, str):
                    found.add(right.value)
    return found


def test_the_supported_set_matches_what_the_dispatcher_implements():
    implemented = _implemented_formats()
    assert implemented, "no export_format branches found -- the guard needs updating"
    assert implemented == set(SUPPORTED_EXPORT_FORMATS)


def test_unknown_formats_are_rejected():
    with pytest.raises(ValueError, match="Unsupported export format"):
        export(_prediction(), "not_a_format", "/nonexistent")


def test_a_compound_format_runs_every_part(monkeypatch, tmp_path):
    """``mini_npz-glb`` is the CLI default, and ``api.inference`` appends
    ``-gs_video`` to it. Each hyphen-separated part must be dispatched."""
    called = []
    for name in ("export_to_mini_npz", "export_to_depth_vis"):
        monkeypatch.setattr(export_mod, name, lambda *a, _n=name, **k: called.append(_n))

    export(_prediction(), "mini_npz-depth_vis", str(tmp_path))

    assert called == ["export_to_mini_npz", "export_to_depth_vis"]


def test_a_compound_format_does_not_fall_through_to_single_handling(monkeypatch, tmp_path):
    """The ``return`` after the loop is load-bearing: without it the last
    part would be exported a second time."""
    called = []
    monkeypatch.setattr(
        export_mod, "export_to_mini_npz", lambda *a, **k: called.append("mini_npz")
    )
    export(_prediction(), "mini_npz-mini_npz", str(tmp_path))
    assert len(called) == 2


def test_per_format_kwargs_reach_only_their_own_writer(monkeypatch, tmp_path):
    seen = {}
    monkeypatch.setattr(export_mod, "export_to_glb", lambda p, d, **k: seen.setdefault("glb", k))
    monkeypatch.setattr(
        export_mod, "export_to_depth_vis", lambda p, d, **k: seen.setdefault("depth_vis", ())
    )

    export(
        _prediction(),
        "glb-depth_vis",
        str(tmp_path),
        glb={"num_max_points": 7},
        depth_vis={"ignored": True},
    )

    assert seen["glb"] == {"num_max_points": 7}
    # export_to_depth_vis takes no kwargs from the table at all.
    assert seen["depth_vis"] == ()


def test_colmap_export_fails_loudly_without_pycolmap():
    """pycolmap is optional, but 'colmap' must never silently no-op.

    The dispatcher advertises 'colmap' in SUPPORTED_EXPORT_FORMATS and
    --export-format is a free string on the CLI, so a swallowed import would
    make `da3 ... --export-format colmap` produce nothing at all.
    """
    import importlib.util

    if importlib.util.find_spec("pycolmap") is not None:
        pytest.skip("pycolmap is installed; the failure path cannot be exercised")

    with pytest.raises(ImportError, match="pycolmap"):
        export(_prediction(), "colmap", "/nonexistent")


def test_the_colmap_error_names_the_extra_that_provides_it():
    import importlib.util

    if importlib.util.find_spec("pycolmap") is not None:
        pytest.skip("pycolmap is installed; the failure path cannot be exercised")

    with pytest.raises(ImportError, match=r"\[colmap\]"):
        export(_prediction(), "colmap", "/nonexistent")


# ---------------------------------------------------------------------------
# npz
# ---------------------------------------------------------------------------
def test_npz_writes_every_field_at_the_documented_path(tmp_path):
    export(_prediction(), "npz", str(tmp_path))
    data = np.load(_wait_for(tmp_path / "exports" / "npz" / "results.npz"))

    assert set(data.files) == {"image", "depth", "conf", "extrinsics", "intrinsics"}
    assert data["image"].shape == (N, H, W, 3) and data["image"].dtype == np.uint8
    assert data["depth"].shape == (N, H, W)
    assert data["extrinsics"].shape == (N, 4, 4)
    assert data["intrinsics"].shape == (N, 3, 3)


def test_npz_values_survive_the_rounding(tmp_path):
    prediction = _prediction()
    export(prediction, "npz", str(tmp_path))
    data = np.load(_wait_for(tmp_path / "exports" / "npz" / "results.npz"))

    # depth is rounded to 6 decimals, conf to 2; poses are stored verbatim.
    np.testing.assert_allclose(data["depth"], prediction.depth, atol=1e-6)
    np.testing.assert_allclose(data["conf"], prediction.conf, atol=5e-3)
    np.testing.assert_array_equal(data["extrinsics"], prediction.extrinsics)
    np.testing.assert_array_equal(data["intrinsics"], prediction.intrinsics)


def test_mini_npz_drops_the_images(tmp_path):
    """That is the whole difference between the two, and it is what makes
    ``mini_npz`` the CLI default."""
    export(_prediction(), "mini_npz", str(tmp_path))
    data = np.load(_wait_for(tmp_path / "exports" / "mini_npz" / "results.npz"))
    assert "image" not in data.files
    assert set(data.files) == {"depth", "conf", "extrinsics", "intrinsics"}


def test_npz_omits_fields_the_prediction_does_not_carry(tmp_path):
    export(_prediction(conf=None, extrinsics=None, intrinsics=None), "mini_npz", str(tmp_path))
    data = np.load(_wait_for(tmp_path / "exports" / "mini_npz" / "results.npz"))
    assert data.files == ["depth"]


@pytest.mark.filterwarnings("ignore::pytest.PytestUnhandledThreadExceptionWarning")
def test_npz_writes_nothing_when_the_prediction_has_no_images(tmp_path):
    """The writer raises inside its worker thread, so nothing propagates to
    the caller -- the warning that escapes is the expected symptom, and is
    filtered here -- but it must also not leave a half-written archive behind."""
    export(_prediction(processed_images=None), "npz", str(tmp_path))
    time.sleep(1.0)
    assert not (tmp_path / "exports" / "npz" / "results.npz").exists()


def test_the_npz_writers_are_fire_and_forget(tmp_path):
    """``@async_call`` starts a thread and returns ``None`` without joining
    it. Callers -- and these tests -- have to wait for the file themselves;
    swapping the decorator for a synchronous one would change that contract
    for every consumer of the exported paths."""
    import threading

    from depth_anything_3.utils.parallel_utils import async_call

    seen = {}
    ready = threading.Event()

    @async_call
    def record():
        seen["thread"] = threading.current_thread()
        ready.set()

    assert record() is None
    assert ready.wait(timeout=10)
    assert seen["thread"] is not threading.current_thread()


# ---------------------------------------------------------------------------
# depth_vis and glb
# ---------------------------------------------------------------------------
def test_depth_vis_writes_one_side_by_side_image_per_view(tmp_path):
    import imageio.v3 as iio

    export(_prediction(), "depth_vis", str(tmp_path))
    files = sorted((tmp_path / "depth_vis").glob("*.jpg"))
    assert [f.name for f in files] == [f"{i:04d}.jpg" for i in range(N)]
    # image and depth are concatenated along the width.
    assert iio.imread(files[0]).shape == (H, 2 * W, 3)


def test_glb_writes_a_scene_trimesh_can_reload(tmp_path):
    import trimesh

    export(_prediction(), "glb", str(tmp_path), glb={"show_cameras": True})

    scene_path = tmp_path / "scene.glb"
    assert scene_path.is_file(), sorted(os.listdir(tmp_path))
    scene = trimesh.load(scene_path)
    assert len(scene.geometry) > 0


def test_glb_point_budget_is_respected(tmp_path):
    import trimesh

    export(_prediction(), "glb", str(tmp_path), glb={"num_max_points": 10, "show_cameras": False})
    scene = trimesh.load(tmp_path / "scene.glb")
    points = sum(
        len(g.vertices) for g in scene.geometry.values() if isinstance(g, trimesh.PointCloud)
    )
    assert 0 < points <= 10
