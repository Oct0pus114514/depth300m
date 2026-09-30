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
def _missing(fmt):
    # [depth300m] heavy export backends are optional for mono-depth use
    def f(*a, **k):
        raise ImportError(
            f"export format '{fmt}' needs extra dependencies; see README (Known limitations)")
    return f

try:
    from depth_anything_3.utils.export.gs import export_to_gs_ply, export_to_gs_video
except ImportError:
    export_to_gs_ply = _missing("gs_ply"); export_to_gs_video = _missing("gs_video")
try:
    from .colmap import export_to_colmap
except ImportError:
    export_to_colmap = _missing("colmap")
try:
    from .glb import export_to_glb
except ImportError:
    export_to_glb = _missing("glb")
from .depth_vis import export_to_depth_vis
from .feat_vis import export_to_feat_vis
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
        export_to_colmap(prediction, export_dir, **kwargs.get(export_format, {}))
    else:
        raise ValueError(f"Unsupported export format: {export_format}")


__all__ = [
    export,
]
