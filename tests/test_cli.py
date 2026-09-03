"""The CLI surface in ``cli.py``, exercised by running it.

Two different jobs here.

*Parser construction* -- typer resolves every type annotation at runtime to
build its parsers, so invoking ``--help`` on each command exercises all 112
``typer.Option`` declarations. That is the standing guard for the deferred
UP007/UP045 migration (see the backlog in CLAUDE.md): if PEP 604 unions ever
break typer's introspection, these fail.

*Option delivery* -- ``--help`` succeeding proves nothing about whether an
option reaches the inference call. Each command is therefore invoked for real
with ``run_inference`` swapped out, and the recorded keyword arguments are
checked. This is what catches an option that is parsed, echoed and then
dropped; ``test_wiring.py`` covers the same boundary statically, and the two
fail on different mistakes.
"""

import numpy as np
import pytest
from PIL import Image
from typer.testing import CliRunner

from depth_anything_3 import cli as cli_module
from depth_anything_3.cli import app, detect_input_type
from depth_anything_3.services.input_handlers import parse_export_feat

COMMANDS = ["auto", "image", "images", "colmap", "video"]


@pytest.fixture(scope="module")
def runner():
    return CliRunner()


@pytest.fixture
def recorded(monkeypatch):
    """Replace ``run_inference`` and collect the keywords it was called with."""
    calls = []
    monkeypatch.setattr(cli_module, "run_inference", lambda **kwargs: calls.append(kwargs))
    return calls


@pytest.fixture
def one_image(tmp_path):
    path = tmp_path / "frame.png"
    Image.fromarray(np.zeros((16, 16, 3), dtype=np.uint8)).save(path)
    return path


# ---------------------------------------------------------------------------
# parser construction
# ---------------------------------------------------------------------------
def test_top_level_help(runner):
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0


@pytest.mark.parametrize("command", COMMANDS)
def test_subcommand_help(runner, command):
    result = runner.invoke(app, [command, "--help"])
    assert result.exit_code == 0, result.output


@pytest.mark.parametrize("command", COMMANDS)
def test_every_command_is_reachable_by_name(runner, command):
    """A command missing its ``@app.command()`` decorator still imports fine."""
    result = runner.invoke(app, ["--help"])
    assert command in result.output


# ---------------------------------------------------------------------------
# input type detection
# ---------------------------------------------------------------------------
def test_detects_a_single_image(tmp_path, one_image):
    assert detect_input_type(str(one_image)) == "image"


def test_detects_a_directory_of_images(tmp_path, one_image):
    assert detect_input_type(str(tmp_path)) == "images"


def test_detects_a_video_by_extension(tmp_path):
    path = tmp_path / "clip.mp4"
    path.write_bytes(b"not really a video")
    assert detect_input_type(str(path)) == "video"


def test_colmap_layout_wins_over_the_images_it_contains(tmp_path):
    """A COLMAP directory *also* contains images; the sparse/ sibling is what
    distinguishes it, and it has to be checked first."""
    (tmp_path / "images").mkdir()
    (tmp_path / "sparse").mkdir()
    Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(tmp_path / "images" / "a.png")
    assert detect_input_type(str(tmp_path)) == "colmap"


@pytest.mark.parametrize("name", ["missing.png", "notes.txt"])
def test_unrecognised_inputs_are_unknown(tmp_path, name):
    path = tmp_path / name
    if name.endswith(".txt"):
        path.write_text("hello")
    assert detect_input_type(str(path)) == "unknown"


def test_an_empty_directory_is_unknown(tmp_path):
    (tmp_path / "empty").mkdir()
    assert detect_input_type(str(tmp_path / "empty")) == "unknown"


# ---------------------------------------------------------------------------
# option parsing helpers
# ---------------------------------------------------------------------------
def test_export_feat_parses_a_comma_separated_list():
    assert parse_export_feat("0,1, 2") == [0, 1, 2]


def test_export_feat_defaults_to_no_layers():
    assert parse_export_feat("") == []


def test_export_feat_rejects_nonsense():
    import typer

    with pytest.raises(typer.BadParameter):
        parse_export_feat("0,one")


# ---------------------------------------------------------------------------
# option delivery
# ---------------------------------------------------------------------------
def test_image_command_forwards_its_options(runner, recorded, tmp_path, one_image):
    result = runner.invoke(
        app,
        [
            "image",
            str(one_image),
            "--export-dir",
            str(tmp_path / "out"),
            "--export-format",
            "npz",
            "--device",
            "cpu",
            "--process-res",
            "112",
            "--export-feat",
            "1,4",
            "--ref-view-strategy",
            "middle",
            "--num-max-points",
            "500",
            "--no-show-cameras",
        ],
    )
    assert result.exit_code == 0, result.output
    (call,) = recorded
    assert call["image_paths"] == [str(one_image)]
    assert call["export_format"] == "npz"
    assert call["device"] == "cpu"
    assert call["process_res"] == 112
    assert call["export_feat_layers"] == [1, 4]
    assert call["ref_view_strategy"] == "middle"
    assert call["num_max_points"] == 500
    assert call["show_cameras"] is False


def test_the_backend_url_is_only_sent_when_use_backend_is_set(
    runner, recorded, tmp_path, one_image
):
    def invoke(name, *extra):
        # A fresh export directory each time: an existing one triggers a prompt.
        args = ["image", str(one_image), "--export-dir", str(tmp_path / name), *extra]
        result = runner.invoke(app, args)
        assert result.exit_code == 0, result.output
        return recorded[-1]["backend_url"]

    assert invoke("a") is None
    assert invoke("b", "--use-backend") == "http://localhost:8008"
    assert invoke("c", "--use-backend", "--backend-url", "http://elsewhere") == "http://elsewhere"


def test_images_command_collects_the_requested_extensions(runner, recorded, tmp_path):
    source = tmp_path / "scene"
    source.mkdir()
    for name in ("a.png", "b.jpg", "c.bmp"):
        Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(source / name)

    result = runner.invoke(
        app,
        [
            "images",
            str(source),
            "--image-extensions",
            "png,jpg",
            "--export-dir",
            str(tmp_path / "o"),
        ],
    )

    assert result.exit_code == 0, result.output
    assert [p.rsplit("/", 1)[-1] for p in recorded[-1]["image_paths"]] == ["a.png", "b.jpg"]


def test_auto_dispatches_a_single_image_like_the_image_command(
    runner, recorded, tmp_path, one_image
):
    result = runner.invoke(app, ["auto", str(one_image), "--export-dir", str(tmp_path / "out")])
    assert result.exit_code == 0, result.output
    assert recorded[-1]["image_paths"] == [str(one_image)]


def test_auto_dispatches_a_colmap_directory_with_its_cameras(
    runner, recorded, monkeypatch, tmp_path
):
    scene = tmp_path / "scene"
    (scene / "images").mkdir(parents=True)
    (scene / "sparse").mkdir()
    extrinsics = np.tile(np.eye(4), (2, 1, 1))
    intrinsics = np.tile(np.eye(3), (2, 1, 1))

    class _Fake:
        @staticmethod
        def process(colmap_dir, sparse_subdir=""):
            assert sparse_subdir == "0"
            return ["a.png", "b.png"], extrinsics, intrinsics

    monkeypatch.setattr(cli_module, "ColmapHandler", _Fake)

    result = runner.invoke(
        app,
        [
            "auto",
            str(scene),
            "--sparse-subdir",
            "0",
            "--export-dir",
            str(tmp_path / "out"),
            "--no-align-to-input-ext-scale",
        ],
    )

    assert result.exit_code == 0, result.output
    call = recorded[-1]
    assert call["image_paths"] == ["a.png", "b.png"]
    np.testing.assert_array_equal(call["extrinsics"], extrinsics)
    np.testing.assert_array_equal(call["intrinsics"], intrinsics)
    assert call["align_to_input_ext_scale"] is False


def test_auto_passes_the_sampling_fps_to_the_video_handler(
    runner, recorded, monkeypatch, tmp_path
):
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"stub")
    seen = {}

    class _Fake:
        @staticmethod
        def process(video_path, output_dir, fps=1.0):
            seen["fps"] = fps
            return ["frame_0.png"]

    monkeypatch.setattr(cli_module, "VideoHandler", _Fake)

    result = runner.invoke(
        app, ["auto", str(clip), "--fps", "3.5", "--export-dir", str(tmp_path / "out")]
    )

    assert result.exit_code == 0, result.output
    assert seen["fps"] == 3.5
    assert recorded[-1]["image_paths"] == ["frame_0.png"]


def test_auto_refuses_an_input_it_cannot_classify(runner, recorded, tmp_path):
    result = runner.invoke(app, ["auto", str(tmp_path / "nothing-here")])
    assert result.exit_code == 1
    assert not recorded


# ---------------------------------------------------------------------------
# export directory handling
# ---------------------------------------------------------------------------
def test_auto_cleanup_empties_an_existing_export_directory(runner, recorded, tmp_path, one_image):
    export_dir = tmp_path / "out"
    export_dir.mkdir()
    stale = export_dir / "stale.txt"
    stale.write_text("from a previous run")

    result = runner.invoke(
        app, ["image", str(one_image), "--export-dir", str(export_dir), "--auto-cleanup"]
    )

    assert result.exit_code == 0, result.output
    assert not stale.exists()
    assert export_dir.is_dir()


def test_declining_the_cleanup_prompt_aborts_without_running(
    runner, recorded, tmp_path, one_image
):
    export_dir = tmp_path / "out"
    export_dir.mkdir()
    keep = export_dir / "keep.txt"
    keep.write_text("precious")

    result = runner.invoke(
        app, ["image", str(one_image), "--export-dir", str(export_dir)], input="n\n"
    )

    assert result.exit_code == 0
    assert keep.exists(), "an existing export directory must not be deleted without consent"
    assert not recorded, "inference must not run after the user declines"


def test_a_missing_image_is_reported_rather_than_forwarded(runner, recorded, tmp_path):
    result = runner.invoke(app, ["image", str(tmp_path / "absent.png")])
    assert result.exit_code != 0
    assert not recorded
