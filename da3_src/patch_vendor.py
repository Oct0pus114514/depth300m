"""One-shot patcher: guard heavy optional imports in the vendored DA3 copy.

The mono-depth path never calls these utilities (GS / multi-view / mesh-video
export only), so mono users don't need pycolmap / evo / trimesh / moviepy.
Applied on top of the pristine Apache-2.0 sources; marked with [depth300m].
"""
from pathlib import Path

ROOT = Path("/root/autodl-tmp/depth300m/da3_src/depth_anything_3")


def patch(rel, old, new):
    p = ROOT / rel
    s = p.read_text()
    assert old in s, f"pattern not found in {rel}: {old[:60]!r}"
    p.write_text(s.replace(old, new, 1))
    print("patched:", rel)


# 1) pose_align: evo is only used inside align_poses_umeyama (multi-view path)
patch("utils/pose_align.py",
      "from evo.core.trajectory import PosePath3D\n",
      "# [depth300m] evo imported lazily below: only the multi-view alignment needs it\n")
patch("utils/pose_align.py",
      "    path_ref = PosePath3D(poses_se3=pose_ref.copy())",
      "    from evo.core.trajectory import PosePath3D  # [depth300m] lazy\n"
      "    path_ref = PosePath3D(poses_se3=pose_ref.copy())")

# 2) export dispatcher: guard the heavy format backends, keep light ones eager
GUARD = '''def _missing(fmt):
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
'''
patch("utils/export/__init__.py",
      "from depth_anything_3.utils.export.gs import export_to_gs_ply, export_to_gs_video\n"
      "\n"
      "from .colmap import export_to_colmap\n"
      "from .depth_vis import export_to_depth_vis\n"
      "from .feat_vis import export_to_feat_vis\n"
      "from .glb import export_to_glb\n"
      "from .npz import export_to_mini_npz, export_to_npz\n",
      GUARD +
      "from .depth_vis import export_to_depth_vis\n"      # imageio/matplotlib: kept, light
      "from .feat_vis import export_to_feat_vis\n"
      "from .npz import export_to_mini_npz, export_to_npz\n")
print("done")
