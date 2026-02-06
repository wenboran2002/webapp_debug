import copy
import os
import sys
from pathlib import Path

import numpy as np
import open3d as o3d
import smplx
import torch


ROOT_DIR = Path(__file__).resolve().parent.parent  # webapp_debug


def resource_path(relative_path: str) -> str:
    """Resolve bundled resource paths.

    The upstream code expected paths like `video_optimizer/data/...` and
    `video_optimizer/smpl_models/SMPLX_NEUTRAL.npz`. We map these to the
    current project layout:
        - data → solver/data
        - smpl model → asset/data/SMPLX_NEUTRAL.npz (shared with the app)
    """

    rel = relative_path
    if rel.startswith("video_optimizer/data"):
        rel = rel.replace("video_optimizer/data", "solver/data")
    if rel.endswith("video_optimizer/smpl_models/SMPLX_NEUTRAL.npz") or rel.startswith(
        "video_optimizer/smpl_models"
    ):
        rel = "asset/data/SMPLX_NEUTRAL.npz"
    return str(ROOT_DIR / rel)


_model_type = "smplx"
_model_folder = resource_path("video_optimizer/smpl_models/SMPLX_NEUTRAL.npz")
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = smplx.create(
    _model_folder,
    model_type=_model_type,
    gender="neutral",
    num_betas=10,
    num_expression_coeffs=10,
    use_pca=False,
    flat_hand_mean=True,
).to(_device)


def apply_initial_transform_to_mesh(mesh: o3d.geometry.TriangleMesh, t: np.ndarray):
    mesh_copy = copy.deepcopy(mesh)
    verts = np.asarray(mesh_copy.vertices)
    transformed_verts = verts + t
    mesh_copy.vertices = o3d.utility.Vector3dVector(transformed_verts)
    return mesh_copy


def apply_initial_transform_to_points(points: np.ndarray, t: np.ndarray) -> np.ndarray:
    return points + t
