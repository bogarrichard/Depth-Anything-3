# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from depth_anything_3.specs import Prediction
from depth_anything_3.utils.export.gs import export_to_gs_ply, export_to_gs_video

from .depth_vis import export_to_depth_vis
from .feat_vis import export_to_feat_vis
from .glb import export_to_glb
from .npz import export_to_mini_npz, export_to_npz

# Canonical set of export formats this dispatcher understands. Treat this as the
# server-trusted enum for any request-facing validation (e.g. the HTTP backend) instead
# of letting a client-supplied export_format string reach the dispatcher unchecked.
SUPPORTED_EXPORT_FORMATS = frozenset(
    {
        "glb",
        "mini_npz",
        "npz",
        "feat_vis",
        "depth_vis",
        "gs_ply",
        "gs_video",
        "colmap",
    }
)


def export(
    prediction: Prediction,
    export_format: str,
    export_dir: str,
    **kwargs,
):
    """Dispatch a prediction to one or more exporters by format name.

    Called by :meth:`depth_anything_3.api.DepthAnything3.inference` when
    ``export_dir`` is given. ``export_format`` may combine several of
    :data:`SUPPORTED_EXPORT_FORMATS` with ``-`` (e.g. ``"mini_npz-glb"``) to
    run each exporter in turn on the same prediction.

    Args:
        prediction: The :class:`~depth_anything_3.specs.Prediction` to
            export.
        export_format: One of :data:`SUPPORTED_EXPORT_FORMATS`, or several
            joined with ``-``. See that format's ``export_to_*`` function
            (:mod:`depth_anything_3.utils.export`) for its own parameters,
            passed via ``kwargs[export_format]``.
        export_dir: Directory to write the exported files into.
        **kwargs: Per-format extra arguments, keyed by format name -- e.g.
            ``kwargs["glb"] = {"num_max_points": 500_000}``.

    Raises:
        ValueError: If ``export_format`` (after splitting on ``-``) is not
            in :data:`SUPPORTED_EXPORT_FORMATS`.
    """
    if "-" in export_format:
        export_formats = export_format.split("-")
        for export_format in export_formats:
            export(prediction, export_format, export_dir, **kwargs)
        return  # Prevent falling through to single-format handling

    if export_format == "glb":
        export_to_glb(prediction, export_dir, **kwargs.get(export_format, {}))
    elif export_format == "mini_npz":
        export_to_mini_npz(prediction, export_dir)
    elif export_format == "npz":
        export_to_npz(prediction, export_dir)
    elif export_format == "feat_vis":
        export_to_feat_vis(prediction, export_dir, **kwargs.get(export_format, {}))
    elif export_format == "depth_vis":
        export_to_depth_vis(prediction, export_dir)
    elif export_format == "gs_ply":
        export_to_gs_ply(prediction, export_dir, **kwargs.get(export_format, {}))
    elif export_format == "gs_video":
        export_to_gs_video(prediction, export_dir, **kwargs.get(export_format, {}))
    elif export_format == "colmap":
        # Imported lazily so pycolmap stays optional: it is a hard-to-build wheel
        # needed by this one format. Must stay an import (not a guarded no-op) so
        # `--export-format colmap` fails loudly rather than silently writing nothing.
        try:
            from .colmap import export_to_colmap
        except ImportError as exc:  # pragma: no cover - depends on the install
            raise ImportError(
                "Export format 'colmap' requires pycolmap. "
                'Install it with: pip install "depth-anything-3[colmap]"'
            ) from exc
        export_to_colmap(prediction, export_dir, **kwargs.get(export_format, {}))
    else:
        raise ValueError(f"Unsupported export format: {export_format}")


__all__ = [
    export,
]
