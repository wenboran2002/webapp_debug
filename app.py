import os
import sys
import cv2
import json
import argparse
import traceback
import numpy as np
from pathlib import Path
from flask import Flask, render_template, request, jsonify, send_file, send_from_directory
import open3d as o3d
from io import BytesIO
from threading import Lock
from copy import deepcopy

try:
    import yaml
except ImportError:
    yaml = None
try:
    from solver.kp_use_new import kp_use_new
except ImportError:
    print("Could not import kp_use_new")
    kp_use_new = None

# Add CoTracker to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'co-tracker'))

try:
    import torch
    import smplx
    from cotracker.predictor import CoTrackerOnlinePredictor
    COTRACKER_AVAILABLE = True
    print("CoTracker and SMPLX imported successfully")
except ImportError as e:
    COTRACKER_AVAILABLE = False
    print(f"Dependency missing: {e}")

app = Flask(__name__, static_folder='static')

# 从 config.yaml 加载常量（缺失时使用默认值）
def _load_config():
    defaults = {
        'mesh': {
            'obj_decimation_target_faces': 30000,
            'obj_decimate_if_vertices_above': 5000,
        },
        'server': {'host': '0.0.0.0', 'port': 5010},
        'video': {'default_fps': 30},
    }
    config_path = Path(__file__).resolve().parent / 'config.yaml'
    if yaml is not None and config_path.exists():
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                loaded = yaml.safe_load(f)
            if isinstance(loaded, dict):
                for k, v in defaults.items():
                    if k in loaded and isinstance(loaded[k], dict):
                        defaults[k] = {**defaults[k], **loaded[k]}
                    elif k in loaded:
                        defaults[k] = loaded[k]
        except Exception as e:
            print(f"Warning: could not load config.yaml: {e}")
    return defaults

CONFIG = _load_config()

# Global variables
VIDEO_PATH = ""
OBJ_PATH = ""
CAP = None
CAP_LOCK = Lock()
MESH_DATA = None # {vertices: [], faces: []}
VIDEO_FRAMES = [] # List of RGB numpy arrays
VIDEO_FRAMES_ENCODED = [] # List of pre-encoded JPEG bytes
VIDEO_FPS = CONFIG['video']['default_fps']
VIDEO_TOTAL_FRAMES = 0
COTRACKER_MODEL = None
COTRACKER_LOCK = Lock()
TRACKED_POINTS = {}
SCENE_DATA = None

# --- HOI 标注任务 / upload_records 管理 ---

# 4d_preprocess_debug 目录（make_hoi.py 所在目录）
PREPROCESS_DIR = Path(__file__).resolve().parent.parent / "4dhoi_autorecon"
UPLOAD_RECORDS_PATH = PREPROCESS_DIR / "upload_records.json"

# 当前这台机器“锁定”的待标注任务（annotation_progress: 2.0 -> 2.1）
HOI_TASKS = []


def _resolve_session_path(p: str) -> Path:
    """
    将 upload_records.json 里的 session_folder 解析为绝对路径。
    行为尽量与 4d_preprocess_debug/make_hoi.py 中 _resolve_path 保持一致。
    """
    if not isinstance(p, str) or not p:
        return Path("")
    if p.startswith("./"):
        return (PREPROCESS_DIR / p[2:]).resolve()
    if p.startswith("tiktok_data/"):
        return (PREPROCESS_DIR / p).resolve()
    return Path(p).expanduser().resolve()


def _load_upload_records() -> list:
    if not UPLOAD_RECORDS_PATH.exists():
        raise FileNotFoundError(f"upload_records.json not found: {UPLOAD_RECORDS_PATH}")
    with UPLOAD_RECORDS_PATH.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("upload_records.json must be a list")
    return data


def _write_upload_records(records: list) -> None:
    tmp_path = UPLOAD_RECORDS_PATH.with_suffix(".json.tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(records, f, indent=4, ensure_ascii=False)
    tmp_path.replace(UPLOAD_RECORDS_PATH)


def _init_hoi_tasks() -> None:
    """
    启动 app 时：
    - 清理历史遗留的 2.1 锁（根据是否已有 kp_record_merged.json 决定回退为 2.0 或升级为 3.0）
    - 读取 upload_records.json
    - 找出 annotation_progress == 2.0 的记录，加入 HOI_TASKS（仅作展示，不修改进度）
    """
    global HOI_TASKS
    if not UPLOAD_RECORDS_PATH.exists():
        print(f"upload_records.json not found at {UPLOAD_RECORDS_PATH}, skip HOI task init")
        return

    # 先清理历史 2.1 锁
    cleaned = _reset_unfinished_hoi_locks()
    if cleaned > 0:
        print(f"Reset {cleaned} unfinished HOI locks from 2.1 to 2.0/3.0")

    try:
        records = _load_upload_records()
    except Exception as e:
        print(f"Failed to load upload_records.json: {e}")
        return

    tasks = []
    for rec in records:
        try:
            prog = float(rec.get("annotation_progress", 0))
        except Exception:
            prog = 0.0
        if prog == 2.0:
            tasks.append(rec)

    HOI_TASKS = tasks


def _update_hoi_progress_for_video_dir(video_dir: str, finished: bool) -> None:
    """
    根据 video_dir 更新对应记录的 annotation_progress：
    - 若 finished=True  -> 3.0（标注完成）
    - 若 finished=False -> 2.0（未标，恢复可被再次领取）
    匹配逻辑：以解析后的绝对路径比较 session_folder 与 video_dir。
    """
    if not UPLOAD_RECORDS_PATH.exists():
        return

    try:
        records = _load_upload_records()
    except Exception as e:
        print(f"Failed to load upload_records.json when updating progress: {e}")
        return

    target_path = Path(video_dir).resolve()
    changed = False

    for rec in records:
        sf = rec.get("session_folder", "")
        if not sf:
            continue
        sf_path = _resolve_session_path(sf)
        if sf_path == target_path:
            rec["annotation_progress"] = 3.0 if finished else 2.0
            changed = True
            break

    if changed:
        try:
            _write_upload_records(records)
            print(
                f"Set annotation_progress to {3.0 if finished else 2.0} "
                f"for session_folder matching {target_path}"
            )
        except Exception as e:
            print(f"Failed to write upload_records.json when updating progress: {e}")


def _reset_unfinished_hoi_locks() -> int:
    """
    将所有 annotation_progress == 2.1 的记录检查一遍：
    - 若 session_folder 下已存在 kp_record_merged.json -> 置为 3.0
    - 否则 -> 置为 2.0
    返回本次修改的记录数量。
    """
    if not UPLOAD_RECORDS_PATH.exists():
        return 0

    try:
        records = _load_upload_records()
    except Exception as e:
        print(f"Failed to load upload_records.json when resetting locks: {e}")
        return 0

    changed = False
    updated_count = 0

    for rec in records:
        try:
            prog = float(rec.get("annotation_progress", 0))
        except Exception:
            continue
        if prog != 2.1:
            continue

        sf = rec.get("session_folder", "")
        if not sf:
            continue
        sf_path = _resolve_session_path(sf)
        kp_merged = sf_path / "kp_record_merged.json"
        if kp_merged.exists():
            rec["annotation_progress"] = 3.0
        else:
            rec["annotation_progress"] = 2.0
        changed = True
        updated_count += 1

    if changed:
        try:
            _write_upload_records(records)
        except Exception as e:
            print(f"Failed to write upload_records.json when resetting locks: {e}")
            return 0

    return updated_count


def _load_video_session(video_dir: str) -> bool:
    """
    根据给定的 video_dir 加载对应的视频与场景数据：
    - 设置 VIDEO_PATH / OBJ_PATH
    - 打开 CAP 并加载所有帧到内存
    - 初始化 CoTracker
    - 构建 SCENE_DATA 并加载 SMPL-X / motion / obj_poses / meshes
    - 准备 MESH_DATA 供前端初始 3D 视图使用
    """
    global VIDEO_PATH, OBJ_PATH, CAP, MESH_DATA, SCENE_DATA, VIDEO_FRAMES, VIDEO_FRAMES_ENCODED, VIDEO_FPS, VIDEO_TOTAL_FRAMES

    video_dir = str(video_dir)
    VIDEO_PATH = os.path.join(video_dir, "video.mp4")
    OBJ_PATH = os.path.join(video_dir, "obj_org.obj")

    # 释放旧的视频句柄
    if CAP is not None:
        try:
            CAP.release()
        except Exception:
            pass
    CAP = None
    VIDEO_FRAMES = []
    VIDEO_FRAMES_ENCODED = []
    VIDEO_FPS = CONFIG['video']['default_fps']
    VIDEO_TOTAL_FRAMES = 0
    MESH_DATA = None
    SCENE_DATA = None

    if not os.path.exists(VIDEO_PATH):
        print(f"Warning: Video not found at {VIDEO_PATH}")
        return False

    CAP = cv2.VideoCapture(VIDEO_PATH)
    if not CAP.isOpened():
        print(f"Failed to open video: {VIDEO_PATH}")
        CAP = None
        return False

    print(f"Video loaded: {VIDEO_PATH}")
    load_video_frames()
    init_cotracker()

    # Initialize Scene Data
    SCENE_DATA = SceneData(video_dir)
    SCENE_DATA.load()

    if os.path.exists(OBJ_PATH) and SCENE_DATA is not None and SCENE_DATA.obj_mesh_org is not None:
        MESH_DATA = load_mesh(SCENE_DATA.obj_mesh_org)
    else:
        print(f"Warning: Mesh not found or obj_mesh_org is None for video_dir {video_dir}")

    return True

def preprocess_obj_sample(obj_org, object_poses, seq_length):
    """Preprocess object mesh for all frames, following app_new.py logic"""
    centers = np.array(object_poses.get('center', []))
    if len(centers) == 0:
        # If no centers provided, create zero centers
        centers = np.zeros((seq_length, 3))
    elif len(centers) < seq_length:
        # Pad with last center if not enough centers
        last_center = centers[-1] if len(centers) > 0 else np.zeros(3)
        centers = np.vstack([centers, np.tile(last_center, (seq_length - len(centers), 1))])
    
    obj_orgs = []
    center_objs = []
    scale = object_poses.get('scale', 1.0)
    
    for i in range(seq_length):
        obj_pcd = deepcopy(obj_org)
        if 'rotation' in object_poses and i < len(object_poses['rotation']):
            rotation_matrix = np.array(object_poses['rotation'][i])
            if rotation_matrix.shape == (3, 3):
                transform_matrix = np.eye(4)
                transform_matrix[:3, :3] = rotation_matrix
                obj_pcd.transform(transform_matrix)
        
        new_overts = np.asarray(obj_pcd.vertices)
        new_overts *= scale
        new_overts = new_overts - np.mean(new_overts, axis=0)
        center_objs.append(np.mean(new_overts, axis=0))
        obj_pcd.vertices = o3d.utility.Vector3dVector(new_overts)
        obj_orgs.append(obj_pcd)
    
    return obj_orgs, centers, center_objs


class SceneData:
    def __init__(self, video_dir):
        self.video_dir = video_dir
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.smplx_model = None
        self.motion_data = None
        self.obj_poses = None
        self.obj_mesh_org = None
        self.obj_mesh_raw = None  # Original high-res mesh
        self.obj_orgs = []  # Preprocessed object meshes for all frames (empty list by default)
        self.obj_orgs_world = [] # Cached world-space meshes for visualization
        self.t_finals = None  # Translation for each frame
        self.R_finals = []  # Rotation matrices for each frame (empty list by default)
        self.hand_poses = None
        self.total_frames = 0
        self.loaded = False
        # Backup of per-frame object vertices for rescaling (always scale from these)
        self.obj_orgs_base_vertices = None
        # Effective object scale factor (after any user rescaling)
        self.object_scale_factor = 1.0

    def update_world_meshes(self):
        """Update cached world-space meshes based on current local meshes and transforms."""
        self.obj_orgs_world = []
        for i, obj in enumerate(self.obj_orgs):
            # Deepcopy to avoid modifying the local mesh
            world_obj = deepcopy(obj)
            verts = np.asarray(world_obj.vertices)
            
            # Get transforms
            if len(self.R_finals) > i:
                R = self.R_finals[i]
            else:
                R = np.eye(3)
                
            if self.t_finals is not None and i < len(self.t_finals):
                t = np.array(self.t_finals[i])
            else:
                t = np.zeros(3)
                
            # Apply transform: verts @ R.T + t
            verts_transformed = np.matmul(verts, R.T) + t
            world_obj.vertices = o3d.utility.Vector3dVector(verts_transformed)
            self.obj_orgs_world.append(world_obj)
        
    def load(self):
        try:
            # Load SMPL-X Model
            model_path = os.path.join(app.root_path, 'asset', 'data', 'SMPLX_NEUTRAL.npz')
            if not os.path.exists(model_path):
                # Try alternative path
                model_path = os.path.join(os.path.dirname(__file__), 'asset', 'data', 'SMPLX_NEUTRAL.npz')
            if os.path.exists(model_path):
                self.smplx_model = smplx.create(model_path, model_type='smplx',
                                              gender='neutral', num_betas=10,
                                              num_expression_coeffs=10,
                                              use_pca=False, flat_hand_mean=True).to(self.device)
            else:
                print(f"SMPL-X model not found at {model_path}")
            
            # Load Motion Data
            motion_path = os.path.join(self.video_dir, 'motion', 'result_hand.pt')
            if os.path.exists(motion_path):
                self.motion_data = torch.load(motion_path, map_location=self.device)
                # Get total frames from motion data
                params = self.motion_data.get('smpl_params_incam', {})
                if 'body_pose' in params:
                    self.total_frames = len(params['body_pose'])
            else:
                print(f"Motion data not found at {motion_path}")
                
            # Load Hand Poses (optional)
            hand_pose_path = os.path.join(self.video_dir, 'motion', 'hand_pose.json')
            if os.path.exists(hand_pose_path):
                with open(hand_pose_path, 'r') as f:
                    self.hand_poses = json.load(f)
            else:
                self.hand_poses = {}
                print(f"Hand pose file not found at {hand_pose_path}, using defaults")
                
            # Load Object Poses
            # Try align folder first, then fall back to output folder
            obj_pose_paths = [
                os.path.join(self.video_dir, 'align', 'obj_poses.json'),
                os.path.join(self.video_dir, 'output', 'obj_poses.json')
            ]
            obj_pose_path = None
            for path in obj_pose_paths:
                if os.path.exists(path):
                    obj_pose_path = path
                    break
            
            if obj_pose_path:
                with open(obj_pose_path, 'r') as f:
                    self.obj_poses = json.load(f)
                print(f"Loaded object poses from {obj_pose_path}")
            else:
                print(f"Object poses not found in any of these locations: {obj_pose_paths}")
            
            # Load Object Mesh
            obj_mesh_path = os.path.join(self.video_dir, 'obj_org.obj')
            if os.path.exists(obj_mesh_path):
                self.obj_mesh_raw = o3d.io.read_triangle_mesh(obj_mesh_path)
                self.obj_mesh_org = deepcopy(self.obj_mesh_raw)
                # Simplify for performance (params from config.yaml)
                target_faces = CONFIG['mesh']['obj_decimation_target_faces']
                vert_threshold = CONFIG['mesh']['obj_decimate_if_vertices_above']
                if len(self.obj_mesh_org.vertices) > vert_threshold:
                    self.obj_mesh_org = self.obj_mesh_org.simplify_quadric_decimation(target_number_of_triangles=target_faces)
            else:
                print(f"Object mesh not found at {obj_mesh_path}")
            
            # Preprocess object meshes for all frames
            if not self.obj_mesh_org:
                print("Warning: obj_mesh_org is None, cannot preprocess object meshes")
            if not self.obj_poses:
                print("Warning: obj_poses is None, cannot preprocess object meshes")
            if self.total_frames == 0:
                print("Warning: total_frames is 0, cannot preprocess object meshes")
            
            if self.obj_mesh_org and self.obj_poses and self.total_frames > 0:
                try:
                    self.obj_orgs, self.t_finals, _ = preprocess_obj_sample(
                        self.obj_mesh_org, self.obj_poses, self.total_frames
                    )
                    # Backup base vertices after all original transforms (rotation, original scale, centering)
                    self.obj_orgs_base_vertices = [
                        np.asarray(obj.vertices).copy() for obj in self.obj_orgs
                    ]
                    # Initialize R_finals as identity matrices (no rotation by default)
                    self.R_finals = [np.eye(3) for _ in range(self.total_frames)]
                    
                    # Ensure rotation list exists in obj_poses
                    if 'rotation' not in self.obj_poses:
                        self.obj_poses['rotation'] = [np.eye(3).tolist() for _ in range(self.total_frames)]
                    
                    # Initialize world meshes
                    self.update_world_meshes()
                    
                    print(f"Preprocessed {len(self.obj_orgs)} object meshes for {self.total_frames} frames")
                except Exception as e:
                    print(f"Error preprocessing object meshes: {e}")
                    import traceback
                    traceback.print_exc()
                    self.obj_orgs = []
            else:
                print(f"Object preprocessing skipped: obj_mesh_org={self.obj_mesh_org is not None}, "
                      f"obj_poses={self.obj_poses is not None}, total_frames={self.total_frames}")
                self.obj_orgs = []
            
            if self.smplx_model and self.motion_data:
                self.loaded = True
                print(f"Scene data loaded (SMPL-X + Motion, {self.total_frames} frames)")
            else:
                print("Scene data incomplete (Missing model or motion file)")
                
        except Exception as e:
            print(f"Error loading scene data: {e}")
            import traceback
            traceback.print_exc()

    def apply_object_scale(self, scale_factor: float):
        """Rescale object meshes for all frames around their centers.

        Always uses the backed-up base vertices so repeated scaling
        does not accumulate errors.
        """
        if scale_factor <= 0:
            raise ValueError("scale_factor must be > 0")

        if not self.obj_orgs:
            raise RuntimeError("No object meshes loaded to scale")

        # Initialize backup if missing or inconsistent
        if (self.obj_orgs_base_vertices is None or
                len(self.obj_orgs_base_vertices) != len(self.obj_orgs)):
            self.obj_orgs_base_vertices = [
                np.asarray(obj.vertices).copy() for obj in self.obj_orgs
            ]

        n_frames = min(self.total_frames, len(self.obj_orgs_base_vertices), len(self.obj_orgs))
        for frame_idx in range(n_frames):
            base_vertices = self.obj_orgs_base_vertices[frame_idx]
            if base_vertices.size == 0:
                continue
            center = np.mean(base_vertices, axis=0)
            vertices_final = (base_vertices - center) * scale_factor + center
            self.obj_orgs[frame_idx].vertices = o3d.utility.Vector3dVector(vertices_final)

        # Update world meshes to reflect new scale
        self.update_world_meshes()

        # Store the user-entered scale factor (relative to the
        # original obj_poses.json scale). This is what we want
        # to save into kp_record_merged.json, matching app_new.py
        # where object_scale comes directly from the UI input.
        self.object_scale_factor = float(scale_factor)

    def get_hand_focus_view(self, frame_idx):
        if not self.loaded:
            return None, "Scene data not loaded"
            
        if frame_idx < 0 or (self.total_frames > 0 and frame_idx >= self.total_frames):
            return None, f"Frame index {frame_idx} out of range"
            
        try:
            # 1. Generate Human Mesh (Logic copied from get_frame_meshes)
            params = self.motion_data['smpl_params_incam']
            
            body_pose = params['body_pose'][frame_idx:frame_idx+1]
            betas = params['betas'][frame_idx:frame_idx+1]
            global_orient = params['global_orient'][frame_idx:frame_idx+1]
            transl = params['transl'][frame_idx:frame_idx+1]
            
            if isinstance(body_pose, torch.Tensor):
                if body_pose.dim() == 3 and body_pose.shape[-1] == 3:
                    body_pose = body_pose.reshape(1, -1)
                elif body_pose.dim() == 2 and body_pose.shape[0] == 1:
                    pass
                elif body_pose.dim() == 1:
                    body_pose = body_pose.unsqueeze(0)
            
            if isinstance(betas, torch.Tensor):
                if betas.dim() == 1:
                    betas = betas.unsqueeze(0)
            
            if isinstance(global_orient, torch.Tensor):
                if global_orient.dim() == 1:
                    global_orient = global_orient.unsqueeze(0)
                elif global_orient.dim() == 3:
                    global_orient = global_orient.squeeze(1)
            
            if isinstance(transl, torch.Tensor):
                if transl.dim() == 1:
                    transl = transl.unsqueeze(0)
            
            left_hand_pose = None
            right_hand_pose = None
            
            if self.hand_poses and str(frame_idx) in self.hand_poses:
                hand_data = self.hand_poses[str(frame_idx)]
                if 'left_hand' in hand_data and hand_data['left_hand'] is not None:
                    left_hand_array = np.array(hand_data['left_hand'])
                    if left_hand_array.size > 0:
                        left_hand_pose = torch.from_numpy(
                            left_hand_array.reshape(-1, 3)[None, ...]
                        ).float().to(self.device)
                if 'right_hand' in hand_data and hand_data['right_hand'] is not None:
                    right_hand_array = np.array(hand_data['right_hand'])
                    if right_hand_array.size > 0:
                        right_hand_pose = torch.from_numpy(
                            right_hand_array.reshape(-1, 3)[None, ...]
                        ).float().to(self.device)
            
            if isinstance(body_pose, torch.Tensor):
                body_pose = body_pose.to(self.device)
            if isinstance(betas, torch.Tensor):
                betas = betas.to(self.device)
            if isinstance(global_orient, torch.Tensor):
                global_orient = global_orient.to(self.device)
            if isinstance(transl, torch.Tensor):
                transl = transl.to(self.device)
            
            zero_pose = torch.zeros((1, 3), dtype=torch.float32, device=self.device)
            
            if left_hand_pose is None:
                left_hand_pose = torch.zeros((1, 15, 3), dtype=torch.float32, device=self.device)
            if right_hand_pose is None:
                right_hand_pose = torch.zeros((1, 15, 3), dtype=torch.float32, device=self.device)
            
            output = self.smplx_model(
                betas=betas,
                global_orient=global_orient,
                body_pose=body_pose,
                left_hand_pose=left_hand_pose,
                right_hand_pose=right_hand_pose,
                jaw_pose=zero_pose,
                leye_pose=zero_pose,
                reye_pose=zero_pose,
                expression=torch.zeros((1, 10), dtype=torch.float32, device=self.device),
                transl=transl,
                return_verts=True
            )
            
            human_verts = output.vertices[0].detach().cpu().numpy()
            joints = output.joints[0].detach().cpu().numpy()
            
            # 2. Get Object Mesh
            if len(self.obj_orgs_world) > 0 and frame_idx < len(self.obj_orgs_world):
                world_obj = self.obj_orgs_world[frame_idx]
                obj_verts = np.asarray(world_obj.vertices)
                obj_center = np.mean(obj_verts, axis=0)
            else:
                # Try to update world meshes if missing
                self.update_world_meshes()
                if len(self.obj_orgs_world) > 0 and frame_idx < len(self.obj_orgs_world):
                    world_obj = self.obj_orgs_world[frame_idx]
                    obj_verts = np.asarray(world_obj.vertices)
                    obj_center = np.mean(obj_verts, axis=0)
                else:
                    return None, "Object mesh not available for this frame"

            # 3. Calculate Focus - SKIPPED (User requested to remove hand magnification)
            
            # Use full mesh
            h_verts = human_verts.tolist()
            h_faces = self.smplx_model.faces.tolist()
            
            # Use full object mesh
            o_verts = obj_verts.tolist()
            o_faces = np.asarray(world_obj.triangles).tolist()

            # 5. Camera Config
            camera = None
            
            return {
                'human': {'vertices': h_verts, 'faces': h_faces},
                'object': {'vertices': o_verts, 'faces': o_faces},
                'camera': camera
            }, None
        except Exception as e:
            print(f"Error generating focus view: {e}")
            traceback.print_exc()
            return None, str(e)

    def get_frame_meshes(self, frame_idx):
        if not self.loaded:
            return None, None
        
        if frame_idx < 0 or (self.total_frames > 0 and frame_idx >= self.total_frames):
            print(f"Frame index {frame_idx} out of range [0, {self.total_frames})")
            return None, None
            
        try:
            # 1. Generate Human Mesh (following app_new.py get_body_points)
            params = self.motion_data['smpl_params_incam']
            
            # Extract parameters for the specific frame
            # Ensure all parameters have batch dimension (first dimension = 1)
            body_pose = params['body_pose'][frame_idx:frame_idx+1]
            betas = params['betas'][frame_idx:frame_idx+1]
            global_orient = params['global_orient'][frame_idx:frame_idx+1]
            transl = params['transl'][frame_idx:frame_idx+1]
            
            # Ensure correct shapes: body_pose should be (1, 63), betas (1, 10), etc.
            # If body_pose is 2D with shape (1, 21, 3), reshape to (1, 63)
            if isinstance(body_pose, torch.Tensor):
                if body_pose.dim() == 3 and body_pose.shape[-1] == 3:
                    # Reshape from (1, 21, 3) to (1, 63)
                    body_pose = body_pose.reshape(1, -1)
                elif body_pose.dim() == 2 and body_pose.shape[0] == 1:
                    # Already correct shape (1, 63)
                    pass
                elif body_pose.dim() == 1:
                    # Add batch dimension: (63) -> (1, 63)
                    body_pose = body_pose.unsqueeze(0)
            
            # Ensure betas has correct shape (1, 10)
            if isinstance(betas, torch.Tensor):
                if betas.dim() == 1:
                    betas = betas.unsqueeze(0)
            
            # Ensure global_orient has correct shape (1, 3)
            if isinstance(global_orient, torch.Tensor):
                if global_orient.dim() == 1:
                    global_orient = global_orient.unsqueeze(0)
                elif global_orient.dim() == 3:
                    # If shape is (1, 1, 3), squeeze middle dimension
                    global_orient = global_orient.squeeze(1)
            
            # Ensure transl has correct shape (1, 3)
            if isinstance(transl, torch.Tensor):
                if transl.dim() == 1:
                    transl = transl.unsqueeze(0)
            
            # Handle hand poses (from hand_pose.json if available)
            # Note: SMPL-X expects all pose inputs to share the same batch size.
            # We therefore force left/right hand poses to have shape (1, 15, 3)
            # so that they are consistent with body_pose/global_orient batch size 1.
            left_hand_pose = None
            right_hand_pose = None
            
            if self.hand_poses and str(frame_idx) in self.hand_poses:
                hand_data = self.hand_poses[str(frame_idx)]
                if 'left_hand' in hand_data and hand_data['left_hand'] is not None:
                    left_hand_array = np.array(hand_data['left_hand'])
                    if left_hand_array.size > 0:
                        # Reshape to (1, num_joints, 3) to match batch size 1
                        left_hand_pose = torch.from_numpy(
                            left_hand_array.reshape(-1, 3)[None, ...]
                        ).float().to(self.device)
                if 'right_hand' in hand_data and hand_data['right_hand'] is not None:
                    right_hand_array = np.array(hand_data['right_hand'])
                    if right_hand_array.size > 0:
                        right_hand_pose = torch.from_numpy(
                            right_hand_array.reshape(-1, 3)[None, ...]
                        ).float().to(self.device)
            
            # Ensure all tensor parameters are on the same device
            if isinstance(body_pose, torch.Tensor):
                body_pose = body_pose.to(self.device)
            if isinstance(betas, torch.Tensor):
                betas = betas.to(self.device)
            if isinstance(global_orient, torch.Tensor):
                global_orient = global_orient.to(self.device)
            if isinstance(transl, torch.Tensor):
                transl = transl.to(self.device)
            
            # Use zero pose for jaw, eyes, and hands if not specified
            # Shape: (batch_size=1, 3) for jaw/eyes, (num_joints, 3) for hands
            zero_pose = torch.zeros((1, 3), dtype=torch.float32, device=self.device)
            
            # For hands, use zero pose with correct shape if not provided
            # SMPL-X has 15 hand joints; we use batch size 1 → (1, 15, 3)
            if left_hand_pose is None:
                left_hand_pose = torch.zeros((1, 15, 3), dtype=torch.float32, device=self.device)
            if right_hand_pose is None:
                right_hand_pose = torch.zeros((1, 15, 3), dtype=torch.float32, device=self.device)
            
            output = self.smplx_model(
                betas=betas,
                global_orient=global_orient,
                body_pose=body_pose,
                left_hand_pose=left_hand_pose,
                right_hand_pose=right_hand_pose,
                jaw_pose=zero_pose,
                leye_pose=zero_pose,
                reye_pose=zero_pose,
                expression=torch.zeros((1, 10), dtype=torch.float32, device=self.device),
                transl=transl,
                return_verts=True
            )
            
            human_verts = output.vertices[0].detach().cpu().numpy()
            human_faces = self.smplx_model.faces.tolist()
            
            # 2. Generate Object Mesh (following app_new.py get_object_points)
            obj_verts = []
            obj_faces = []
            
            # Use cached world-space meshes
            if len(self.obj_orgs_world) > 0 and frame_idx < len(self.obj_orgs_world):
                world_obj = self.obj_orgs_world[frame_idx]
                obj_verts = np.asarray(world_obj.vertices).tolist()
                obj_faces = np.asarray(world_obj.triangles).tolist()

            else:
                print(f"Object mesh not available for frame {frame_idx}: obj_orgs len={len(self.obj_orgs)}, "
                      f"obj_mesh_org={'loaded' if self.obj_mesh_org else 'None'}, "
                      f"obj_poses={'loaded' if self.obj_poses else 'None'}, "
                      f"total_frames={self.total_frames}")
                return None, None
            
            return (human_verts.tolist(), human_faces), (obj_verts, obj_faces)
            
        except Exception as e:
            print(f"Error generating frame meshes for frame {frame_idx}: {e}")
            import traceback
            traceback.print_exc()
            return None, None


def load_mesh(obj_mesh):
    # if not os.path.exists(path):
    #     return None
    
    # print(f"Loading mesh from {path}...")
    # mesh = o3d.io.read_triangle_mesh(path)
    
    # if not mesh.has_vertices():
    #     return None

    # # Downsample for web performance if needed
    # # Target around 10k vertices for smooth interaction
    # if len(mesh.vertices) >20000:
    #     print(f"Downsampling mesh from {len(mesh.vertices)} vertices...")
    #     mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=20000)
    # sample_path=path.replace('.obj', '_sampled.obj')
    # o3d.io.write_triangle_mesh(sample_path, mesh)
    
    vertices = np.asarray(obj_mesh.vertices).tolist()
    faces = np.asarray(obj_mesh.triangles).tolist()
    
    print(f"Mesh loaded: {len(vertices)} vertices, {len(faces)} faces")
    return {'vertices': vertices, 'faces': faces}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/asset/<path:filename>')
def serve_asset(filename):
    return send_from_directory('asset', filename)

@app.route('/api/metadata')
def get_metadata():
    global CAP, VIDEO_TOTAL_FRAMES, VIDEO_FPS
    with CAP_LOCK:
        if CAP is None:
            return jsonify({'error': 'Video not loaded'}), 500

        # Use pre-calculated values if available, otherwise get from CAP
        total_frames = VIDEO_TOTAL_FRAMES or int(CAP.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(CAP.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(CAP.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = VIDEO_FPS or CAP.get(cv2.CAP_PROP_FPS)

    return jsonify({
        'total_frames': total_frames,
        'width': width,
        'height': height,
        'fps': fps,
        'has_mesh': MESH_DATA is not None,
        'video_name': os.path.basename(VIDEO_PATH),
        'obj_name': os.path.basename(OBJ_PATH)
    })


@app.route('/api/hoi_tasks')
def get_hoi_tasks():
    """
    返回当前 annotation_progress == 2.0 的所有 HOI 待标注任务列表。
    每次调用都会从 upload_records.json 重新读取，保证结果实时。
    """
    if not UPLOAD_RECORDS_PATH.exists():
        return jsonify({'tasks': []})

    try:
        records = _load_upload_records()
    except Exception as e:
        print(f"Failed to load upload_records.json in get_hoi_tasks: {e}")
        return jsonify({'tasks': []})

    tasks = []
    for rec in records:
        try:
            prog = float(rec.get("annotation_progress", 0))
        except Exception:
            prog = 0.0
        if prog == 2.0:
            tasks.append(rec)

    return jsonify({'tasks': tasks})


@app.route('/api/hoi_start', methods=['POST'])
def hoi_start():
    """
    开始标注某个 session：
    - 前端提供 session_folder（与 upload_records.json 中一致的字符串）
    - 仅允许从 annotation_progress == 2.0 切换到 2.1
    - 同时在后端加载对应 video_dir（video.mp4 / obj_org.obj / motion 等）
    """
    payload = request.get_json(silent=True) or {}
    session_folder = payload.get('session_folder')
    if not isinstance(session_folder, str) or not session_folder:
        return jsonify({'error': 'session_folder is required'}), 400

    if not UPLOAD_RECORDS_PATH.exists():
        return jsonify({'error': 'upload_records.json not found'}), 500

    try:
        records = _load_upload_records()
    except Exception as e:
        print(f"Failed to load upload_records.json in hoi_start: {e}")
        return jsonify({'error': 'failed to load upload_records.json'}), 500

    target_rec = None
    for rec in records:
        if str(rec.get('session_folder', '')) == session_folder:
            target_rec = rec
            break

    if target_rec is None:
        return jsonify({'error': f'session_folder not found: {session_folder}'}), 404

    try:
        prog = float(target_rec.get('annotation_progress', 0))
    except Exception:
        prog = 0.0

    if prog != 2.0:
        return jsonify({'error': f'annotation_progress must be 2.0 to start, got {prog}'}), 400

    # 解析为绝对路径
    video_dir_path = _resolve_session_path(session_folder)
    if not video_dir_path.exists():
        return jsonify({'error': f'video_dir not exists: {video_dir_path}'}), 404

    # 尝试加载该 session 对应的数据
    ok = _load_video_session(str(video_dir_path))
    if not ok:
        return jsonify({'error': f'failed to load video session at {video_dir_path}'}), 500

    # 若加载成功，再将 progress 2.0 -> 2.1，避免锁死无效样本
    target_rec['annotation_progress'] = 2.1
    try:
        _write_upload_records(records)
    except Exception as e:
        print(f"Failed to write upload_records.json in hoi_start: {e}")
        return jsonify({'error': 'failed to update upload_records.json'}), 500

    return jsonify({
        'status': 'success',
        'video_dir': str(video_dir_path),
        'record': target_rec,
    })


@app.route('/api/hoi_finish', methods=['POST'])
def hoi_finish():
    """
    结束当前标注：
    - 用于“放弃/结束”当前标注但不调用 save_merged_annotations 的场景
    - 将指定 session_folder 的 annotation_progress 从 2.1 改回 2.0
     （如果已经被 save_merged_annotations 改为 3.0，则不会修改）
    """
    payload = request.get_json(silent=True) or {}
    session_folder = payload.get('session_folder')
    if not isinstance(session_folder, str) or not session_folder:
        return jsonify({'error': 'session_folder is required'}), 400

    if not UPLOAD_RECORDS_PATH.exists():
        return jsonify({'error': 'upload_records.json not found'}), 500

    try:
        records = _load_upload_records()
    except Exception as e:
        print(f"Failed to load upload_records.json in hoi_finish: {e}")
        return jsonify({'error': 'failed to load upload_records.json'}), 500

    target_rec = None
    for rec in records:
        if str(rec.get('session_folder', '')) == session_folder:
            target_rec = rec
            break

    if target_rec is None:
        return jsonify({'error': f'session_folder not found: {session_folder}'}), 404

    try:
        prog = float(target_rec.get('annotation_progress', 0))
    except Exception:
        prog = 0.0

    if prog == 2.1:
        target_rec['annotation_progress'] = 2.0
        try:
            _write_upload_records(records)
        except Exception as e:
            print(f"Failed to write upload_records.json in hoi_finish: {e}")
            return jsonify({'error': 'failed to update upload_records.json'}), 500

    return jsonify({'status': 'success', 'annotation_progress': target_rec.get('annotation_progress')})


@app.route('/api/finalize_hoi_sessions', methods=['POST'])
def finalize_hoi_sessions():
    """
    扫描所有 annotation_progress == 2.1 的记录：
    - 若对应 session_folder 下存在 kp_record_merged.json，则改为 3.0
    - 否则改回 2.0
    用于一次性“收尾”，保证未完成的任务不会一直被锁住。
    """
    updated = _reset_unfinished_hoi_locks()
    return jsonify({'status': 'success', 'updated_records': updated})

@app.route('/api/frame/<int:frame_idx>')
def get_frame(frame_idx):
    global VIDEO_FRAMES_ENCODED, VIDEO_TOTAL_FRAMES

    # Clamp frame index
    frame_idx = max(0, min(frame_idx, VIDEO_TOTAL_FRAMES - 1))

    # Check if we have pre-encoded frames
    if VIDEO_FRAMES_ENCODED and len(VIDEO_FRAMES_ENCODED) > frame_idx:
        # Use pre-encoded frame data for instant response
        frame_data = VIDEO_FRAMES_ENCODED[frame_idx]
        io_buf = BytesIO(frame_data)

        # Set cache headers for better performance
        response = send_file(io_buf, mimetype='image/jpeg')
        response.headers['Cache-Control'] = 'public, max-age=31536000'  # Cache for 1 year
        response.headers['ETag'] = f'frame-{frame_idx}'
        return response

    # Fallback to original method if pre-encoded frames not available
    global CAP
    with CAP_LOCK:
        if CAP is None:
            return "Video not loaded", 404

        CAP.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = CAP.read()

    if not ret:
        return "Frame read error", 500

    # Encode to JPEG with optimized quality for faster loading
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, 85]
    ret, buffer = cv2.imencode('.jpg', frame, encode_params)
    io_buf = BytesIO(buffer)

    # Set cache headers to allow browser caching but prevent stale cache
    response = send_file(io_buf, mimetype='image/jpeg')
    response.headers['Cache-Control'] = 'public, max-age=3600'  # Cache for 1 hour
    response.headers['ETag'] = str(frame_idx)  # Use frame index as ETag
    return response

@app.route('/api/mesh')
def get_mesh():
    if MESH_DATA is None:
        return jsonify({'error': 'Mesh not loaded'}), 404
    
    # Prepare data for Plotly
    # x, y, z arrays
    vertices = MESH_DATA['vertices']
    faces = MESH_DATA['faces']
    
    x, y, z = zip(*vertices)
    i, j, k = zip(*faces) if faces else ([], [], [])
    
    return jsonify({
        'x': x, 'y': y, 'z': z,
        'i': i, 'j': j, 'k': k
    })

@app.route('/api/scene_data/<int:frame_idx>')
def get_scene_data(frame_idx):
    if SCENE_DATA is None or not SCENE_DATA.loaded:
        return jsonify({'error': 'Scene data not loaded'}), 404
        
    human, obj = SCENE_DATA.get_frame_meshes(frame_idx)
    
    if human is None or obj is None:
        error_msg = 'Failed to generate meshes'
        if human is None:
            error_msg += ' (human mesh)'
        if obj is None:
            error_msg += ' (object mesh)'
        return jsonify({'error': error_msg}), 500
        
    h_verts, h_faces = human
    o_verts, o_faces = obj
    
    # Prepare for Plotly
    # Human
    hx, hy, hz = zip(*h_verts) if h_verts else ([], [], [])
    hi, hj, hk = zip(*h_faces) if h_faces else ([], [], [])
    
    # Object
    ox, oy, oz = zip(*o_verts) if o_verts else ([], [], [])
    oi, oj, ok = zip(*o_faces) if o_faces else ([], [], [])
    
    return jsonify({
        'human': {
            'x': list(hx), 'y': list(hy), 'z': list(hz),
            'i': list(hi), 'j': list(hj), 'k': list(hk)
        },
        'object': {
            'x': list(ox), 'y': list(oy), 'z': list(oz),
            'i': list(oi), 'j': list(oj), 'k': list(ok)
        }
    })

@app.route('/api/focus_hand/<int:frame_idx>')
def focus_hand(frame_idx):
    if SCENE_DATA is None or not SCENE_DATA.loaded:
        return jsonify({'error': 'Scene data not loaded'}), 404
        
    data, error = SCENE_DATA.get_hand_focus_view(frame_idx)
    if data is None:
        return jsonify({'error': f'Failed to generate focus view: {error}'}), 500
        
    # Prepare for Plotly
    h_verts = data['human']['vertices']
    h_faces = data['human']['faces']
    if h_verts and len(h_verts) > 0:
        hx, hy, hz = zip(*h_verts)
    else:
        hx, hy, hz = [], [], []
        
    if h_faces and len(h_faces) > 0:
        hi, hj, hk = zip(*h_faces)
    else:
        hi, hj, hk = [], [], []
    
    o_verts = data['object']['vertices']
    o_faces = data['object']['faces']
    if o_verts and len(o_verts) > 0:
        ox, oy, oz = zip(*o_verts)
    else:
        ox, oy, oz = [], [], []
        
    if o_faces and len(o_faces) > 0:
        oi, oj, ok = zip(*o_faces)
    else:
        oi, oj, ok = [], [], []
    
    return jsonify({
        'human': {
            'x': list(hx), 'y': list(hy), 'z': list(hz),
            'i': list(hi), 'j': list(hj), 'k': list(hk)
        },
        'object': {
            'x': list(ox), 'y': list(oy), 'z': list(oz),
            'i': list(oi), 'j': list(oj), 'k': list(ok)
        },
        'camera': data['camera']
    })

@app.route('/api/set_scale', methods=['POST'])
def set_scale():
    """Set a new global object scale for scene visualization.

    The scaling is applied on the server side based on the
    original per-frame object geometry stored in SceneData,
    following center-based scaling:

        center = mean(base_vertices)
        vertices_final = (base_vertices - center) * scale_factor + center
    """
    global SCENE_DATA

    if SCENE_DATA is None or not SCENE_DATA.loaded:
        return jsonify({'error': 'Scene data not loaded'}), 404

    data = request.get_json(silent=True) or {}
    scale_factor = data.get('scale_factor')

    try:
        scale_factor = float(scale_factor)
    except (TypeError, ValueError):
        return jsonify({'error': 'Invalid scale_factor'}), 400

    if scale_factor <= 0:
        return jsonify({'error': 'scale_factor must be > 0'}), 400

    try:
        SCENE_DATA.apply_object_scale(scale_factor)
    except Exception as e:
        print(f"Error applying object scale: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

    return jsonify({'status': 'success', 'scale_factor': scale_factor})

@app.route('/api/track_2d', methods=['POST'])
def track_2d():
    global COTRACKER_MODEL, VIDEO_FRAMES, COTRACKER_LOCK
    
    if not COTRACKER_AVAILABLE or COTRACKER_MODEL is None:
        return jsonify({'error': 'CoTracker not available'}), 500
        
    data = request.json
    # Ensure start_frame is an integer index
    try:
        start_frame = int(data.get('frame_idx', 0))
    except (TypeError, ValueError):
        start_frame = 0
    x = data.get('x')
    y = data.get('y')
    
    if x is None or y is None:
        return jsonify({'error': 'Missing coordinates'}), 400
        
    if not VIDEO_FRAMES:
        return jsonify({'error': 'Video frames not loaded in memory'}), 500

    # CoTrackerOnlinePredictor keeps internal online state and is NOT thread-safe.
    # Serialize all tracking requests to avoid race conditions when the
    # frontend issues multiple /api/track_2d calls.
    with COTRACKER_LOCK:
        try:
            device = next(COTRACKER_MODEL.parameters()).device

            queries = [[0.0, x, y]]
            queries_tensor = torch.tensor(queries, dtype=torch.float32).to(device)

            video_sequence = VIDEO_FRAMES[start_frame:]
            if not video_sequence:
                return jsonify({'error': 'No frames to track'}), 400

            window_frames = [video_sequence[0]]

            def _process_step(window_frames, is_first_step, queries=None):
                step = COTRACKER_MODEL.step
                frames_to_use = window_frames[-step * 2:] if len(window_frames) >= step * 2 else window_frames
                if len(frames_to_use) == 0:
                    return None, None

                video_chunk = (
                    torch.tensor(
                        np.stack(frames_to_use), device=device
                    )
                    .float()
                    .permute(0, 3, 1, 2)[None]
                )  # (1, T, 3, H, W)

                return COTRACKER_MODEL(
                    video_chunk,
                    is_first_step=is_first_step,
                    queries=queries,
                    grid_size=0,
                    grid_query_frame=0,
                )

            is_first_step = True
            pred_tracks_list = []
            pred_visibility_list = []

            step = COTRACKER_MODEL.step

            for i in range(1, len(video_sequence)):
                window_frames.append(video_sequence[i])

                if (i % step == 0) or (i == len(video_sequence) - 1):
                    pred_tracks, pred_visibility = _process_step(
                        window_frames,
                        is_first_step,
                        queries=queries_tensor[None] if is_first_step else None,
                    )

                    if pred_tracks is not None:
                        pred_tracks_list.append(pred_tracks)
                        pred_visibility_list.append(pred_visibility)

                    is_first_step = False

            if not pred_tracks_list:
                return jsonify({'error': 'Tracking failed to produce results'}), 500

            final_tracks = pred_tracks_list[-1][0].permute(1, 0, 2).cpu().numpy()  # [num_points, num_frames, 2]

            tracks = final_tracks[0]  # [num_frames, 2]

            # Build result dict only from the requested start_frame to the end
            result = {}
            for i in range(len(tracks)):
                abs_frame = start_frame + i
                if abs_frame < start_frame:
                    continue
                result[abs_frame] = tracks[i].tolist()

            return jsonify({'status': 'success', 'tracks': result})

        except Exception as e:
            print(f"Tracking error: {e}")
            import traceback
            traceback.print_exc()
            return jsonify({'error': str(e)}), 500


@app.route('/api/track_2d_multi', methods=['POST'])
def track_2d_multi():
    """Track multiple 2D points in a single CoTracker run.

    Request JSON format:
        {
            "frame_idx": <int>,
            "points": [
                {"obj_idx": <int>, "x": <float>, "y": <float>},
                ...
            ]
        }

    Response JSON format:
        {
            "status": "success",
            "tracks": {
                "<obj_idx>": {"<frame_idx>": [x, y], ...},
                ...
            }
        }
    """
    global COTRACKER_MODEL, VIDEO_FRAMES, COTRACKER_LOCK

    if not COTRACKER_AVAILABLE or COTRACKER_MODEL is None:
        return jsonify({'error': 'CoTracker not available'}), 500

    data = request.json or {}

    try:
        start_frame = int(data.get('frame_idx', 0))
    except (TypeError, ValueError):
        start_frame = 0

    points = data.get('points') or []
    if not isinstance(points, list) or len(points) == 0:
        return jsonify({'error': 'No points provided for tracking'}), 400

    if not VIDEO_FRAMES:
        return jsonify({'error': 'Video frames not loaded in memory'}), 500

    # Prepare queries and a parallel list of obj indices
    queries = []
    obj_indices = []
    for p in points:
        try:
            x = float(p.get('x'))
            y = float(p.get('y'))
        except (TypeError, ValueError):
            continue
        obj_idx = p.get('obj_idx')
        obj_indices.append(obj_idx)
        queries.append([0.0, x, y])  # use 0 as the first frame within the window

    if len(queries) == 0:
        return jsonify({'error': 'No valid points to track'}), 400

    with COTRACKER_LOCK:
        try:
            device = next(COTRACKER_MODEL.parameters()).device

            queries_tensor = torch.tensor(queries, dtype=torch.float32).to(device)

            video_sequence = VIDEO_FRAMES[start_frame:]
            if not video_sequence:
                return jsonify({'error': 'No frames to track'}), 400

            window_frames = [video_sequence[0]]

            def _process_step(window_frames, is_first_step, queries=None):
                step = COTRACKER_MODEL.step
                frames_to_use = window_frames[-step * 2:] if len(window_frames) >= step * 2 else window_frames
                if len(frames_to_use) == 0:
                    return None, None

                video_chunk = (
                    torch.tensor(
                        np.stack(frames_to_use), device=device
                    )
                    .float()
                    .permute(0, 3, 1, 2)[None]
                )  # (1, T, 3, H, W)

                return COTRACKER_MODEL(
                    video_chunk,
                    is_first_step=is_first_step,
                    queries=queries,
                    grid_size=0,
                    grid_query_frame=0,
                )

            is_first_step = True
            pred_tracks_list = []
            pred_visibility_list = []

            step = COTRACKER_MODEL.step

            for i in range(1, len(video_sequence)):
                window_frames.append(video_sequence[i])

                if (i % step == 0) or (i == len(video_sequence) - 1):
                    pred_tracks, pred_visibility = _process_step(
                        window_frames,
                        is_first_step,
                        queries=queries_tensor[None] if is_first_step else None,
                    )

                    if pred_tracks is not None:
                        pred_tracks_list.append(pred_tracks)
                        pred_visibility_list.append(pred_visibility)

                    is_first_step = False

            if not pred_tracks_list:
                return jsonify({'error': 'Tracking failed to produce results'}), 500

            final_tracks = pred_tracks_list[-1][0].permute(1, 0, 2).cpu().numpy()  # [num_points, num_frames, 2]

            result = {}
            num_points, num_frames, _ = final_tracks.shape
            for i in range(num_points):
                obj_idx = obj_indices[i]
                key = str(obj_idx)
                tracks_i = final_tracks[i]
                frame_dict = {}
                for t in range(num_frames):
                    abs_frame = start_frame + t
                    frame_dict[abs_frame] = tracks_i[t].tolist()
                result[key] = frame_dict

            return jsonify({'status': 'success', 'tracks': result})

        except Exception as e:
            print(f"Multi-point tracking error: {e}")
            import traceback
            traceback.print_exc()
            return jsonify({'error': str(e)}), 500

@app.route('/api/save_annotation', methods=['POST'])
def save_annotation():
    """Save annotation for a single frame.

    This writes a kp_record-style JSON file for the given frame,
    compatible with the desktop app_new.py logic:

        {
            "2D_keypoint": [[obj_idx, [x, y]], ...],
            "joint_name_1": obj_idx_for_joint_1,
            ...
        }

    Only the specified frame is written; callers can invoke this
    multiple times for different frames and later run the desktop
    merge routine (save_kp_record_merged) if needed.
    """
    global SCENE_DATA, VIDEO_PATH

    data = request.get_json(silent=True) or {}

    try:
        frame_idx = int(data.get('frame', 0))
    except (TypeError, ValueError):
        return jsonify({'error': 'Invalid frame index'}), 400

    human_keypoints = data.get('human_keypoints', {}) or {}
    tracks = data.get('tracks', {}) or {}

    # Build kp_record structure
    kp_record = {}

    # 3D keypoints: joint_name -> object point index (int)
    for joint_name, info in human_keypoints.items():
        if not isinstance(info, dict):
            continue
        idx = info.get('index')
        if idx is None:
            continue
        try:
            idx = int(idx)
        except (TypeError, ValueError):
            continue
        kp_record[joint_name] = idx

    # 2D keypoints for this frame only: list of [obj_idx, [x, y]]
    two_d_list = []
    for obj_idx_str, track in tracks.items():
        try:
            obj_idx = int(obj_idx_str)
        except (TypeError, ValueError):
            obj_idx = obj_idx_str

        if not isinstance(track, dict):
            continue

        pt = track.get(str(frame_idx)) or track.get(frame_idx)
        if not pt or len(pt) < 2:
            continue

        try:
            x = float(pt[0])
            y = float(pt[1])
        except (TypeError, ValueError):
            continue

        two_d_list.append([obj_idx, [x, y]])

    kp_record['2D_keypoint'] = two_d_list

    # Determine video directory to place kp_record folder
    video_dir = None
    if SCENE_DATA is not None and getattr(SCENE_DATA, 'video_dir', None):
        video_dir = SCENE_DATA.video_dir
    elif VIDEO_PATH:
        video_dir = os.path.dirname(VIDEO_PATH)

    if not video_dir:
        return jsonify({'error': 'Video directory not available; cannot save annotations'}), 500

    kp_dir = os.path.join(video_dir, 'kp_record')
    os.makedirs(kp_dir, exist_ok=True)

    out_path = os.path.join(kp_dir, f"{frame_idx:05d}.json")
    try:
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(kp_record, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"Error saving annotation for frame {frame_idx} to {out_path}: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

    return jsonify({'status': 'success', 'frame': frame_idx})


@app.route('/api/save_merged_annotations', methods=['POST'])
def save_merged_annotations():
    """Merge per-frame kp_record JSONs into a single kp_record_merged.json.

    This mirrors the desktop save_kp_record_merged logic:
    - Scans kp_record/00000.json .. 00xxx.json
    - Computes DoF = 3 * num_3d + 2 * num_2d for each frame
    - Requires at least one annotated frame and DoF >= 6 for all
      frames from the first annotated frame onward.

    Request JSON (optional):
        { "is_static_object": bool }

    Response JSON on success:
        {
            "status": "success",
            "path": ".../kp_record_merged.json",
            "first_annotated_frame": <int>,
            "last_frame_index": <int>
        }
    """
    global SCENE_DATA, VIDEO_PATH

    payload = request.get_json(silent=True) or {}

    # Determine video directory
    video_dir = None
    if SCENE_DATA is not None and getattr(SCENE_DATA, 'video_dir', None):
        video_dir = SCENE_DATA.video_dir
    elif VIDEO_PATH:
        video_dir = os.path.dirname(VIDEO_PATH)

    if not video_dir:
        return jsonify({'error': 'Video directory not available; cannot merge annotations'}), 500

    # If the frontend provides in-memory annotations (joint keyframes and
    # tracks), prefer building the merged structure from those, similar to
    # how app_new.py maintains per-frame kp_record automatically.
    joint_keyframes = payload.get('joint_keyframes') or {}
    visibility_keyframes = payload.get('visibility_keyframes') or {}
    tracks = payload.get('tracks') or {}
    total_frames = payload.get('total_frames')

    # Optional: limit saving/merging to frames [0, last_frame]
    # so that when the user clicks "Save All" at a certain
    # frame on the timeline, we do not require or save any
    # annotations for later frames.
    last_frame_param = payload.get('last_frame')
    try:
        last_frame = int(last_frame_param) if last_frame_param is not None else None
    except (TypeError, ValueError):
        last_frame = None

    def is_visible_at_frame(obj_idx_str: str, frame_idx: int) -> bool:
        kfs = visibility_keyframes.get(str(obj_idx_str)) or []
        if not kfs:
            return True
        result = True
        for kf in kfs:
            try:
                f = int(kf.get('frame', 0))
            except Exception:
                continue
            if f <= frame_idx:
                result = bool(kf.get('visible', True))
            else:
                break
        return result

    def joint_for_obj_at_frame(obj_idx_str: str, frame_idx: int):
        kfs = joint_keyframes.get(str(obj_idx_str)) or []
        if not kfs:
            return None
        result = None
        for kf in kfs:
            try:
                f = int(kf.get('frame', 0))
            except Exception:
                continue
            if f <= frame_idx:
                result = kf.get('joint')
            else:
                break
        return result

    merged = {}
    invalid_frames = []
    first_annotated_frame = None

    try:
        if joint_keyframes or tracks:
            # Build from in-memory annotations provided by the frontend
            if total_frames is None:
                # Fallback: infer from 2D tracks if total_frames is missing
                candidate_frames = []
                for obj_idx_str, tr in tracks.items():
                    for f_str in tr.keys():
                        try:
                            candidate_frames.append(int(f_str))
                        except Exception:
                            continue
                if candidate_frames:
                    total_frames = max(candidate_frames) + 1
            if total_frames is None:
                return jsonify({'error': 'total_frames is required when using in-memory annotations'}), 400

            total_frames = int(total_frames)

            # Determine the last frame index to include
            if last_frame is not None:
                max_frame = min(int(last_frame), total_frames - 1)
            else:
                max_frame = total_frames - 1

            # Collect all object indices that ever appear
            all_obj_indices = set()
            for k in joint_keyframes.keys():
                all_obj_indices.add(str(k))
            for k in tracks.keys():
                all_obj_indices.add(str(k))

            for frame_idx in range(0, max_frame + 1):
                frame_kp = {}

                # 3D joints: derive joint_name -> obj_idx from keyframes
                # and visibility per frame.
                joint_map = {}
                for obj_idx_str in all_obj_indices:
                    if not is_visible_at_frame(obj_idx_str, frame_idx):
                        continue
                    joint_name = joint_for_obj_at_frame(obj_idx_str, frame_idx)
                    if joint_name:
                        try:
                            obj_idx_int = int(obj_idx_str)
                        except Exception:
                            obj_idx_int = obj_idx_str
                        # Last assignment wins if multiple objects use same joint
                        joint_map[str(joint_name)] = obj_idx_int

                # Write 3D keys into frame_kp
                for joint_name, obj_idx_val in joint_map.items():
                    frame_kp[joint_name] = obj_idx_val

                # 2D keypoints: tracks[obj_idx][frame] -> [x, y]
                two_d_list = []
                for obj_idx_str, tr in tracks.items():
                    pt = tr.get(str(frame_idx)) or tr.get(frame_idx)
                    if not pt or len(pt) < 2:
                        continue
                    try:
                        x = float(pt[0])
                        y = float(pt[1])
                    except (TypeError, ValueError):
                        continue
                    try:
                        obj_idx_int = int(obj_idx_str)
                    except Exception:
                        obj_idx_int = obj_idx_str
                    two_d_list.append([obj_idx_int, [x, y]])

                frame_kp['2D_keypoint'] = two_d_list

                num_2d = len(two_d_list)
                num_3d = len([k for k in frame_kp.keys() if k != '2D_keypoint'])
                has_annotation = (num_2d > 0) or (num_3d > 0)

                if has_annotation and first_annotated_frame is None:
                    first_annotated_frame = frame_idx

                dof = 3 * num_3d + 2 * num_2d
                if first_annotated_frame is not None and frame_idx >= first_annotated_frame and dof < 6:
                    invalid_frames.append((frame_idx, dof, num_3d, num_2d))

                merged[f"{frame_idx:05d}"] = frame_kp

        else:
            # Fallback: merge existing per-frame kp_record JSONs on disk
            kp_dir = os.path.join(video_dir, 'kp_record')
            if not os.path.isdir(kp_dir):
                return jsonify({'error': 'kp_record folder not found; nothing to merge'}), 400

            frame_indices = []
            for fname in os.listdir(kp_dir):
                if not fname.endswith('.json'):
                    continue
                stem = os.path.splitext(fname)[0]
                if len(stem) != 5:
                    continue
                try:
                    idx = int(stem)
                except ValueError:
                    continue
                frame_indices.append(idx)

            if not frame_indices:
                return jsonify({'error': 'No per-frame kp_record JSON files found to merge'}), 400

            max_frame = max(frame_indices)

            # If caller requested an explicit last_frame, respect it
            if last_frame is not None:
                max_frame = min(max_frame, int(last_frame))

            kp_dir = os.path.join(video_dir, 'kp_record')

            for frame_idx in range(0, max_frame + 1):
                fname = f"{frame_idx:05d}.json"
                fpath = os.path.join(kp_dir, fname)
                if not os.path.exists(fpath):
                    # If we already saw an annotated frame, missing later frames are invalid
                    if first_annotated_frame is not None:
                        invalid_frames.append((frame_idx, 0, 0, 0))
                    continue

                with open(fpath, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                num_2d = len(data.get('2D_keypoint', []) or [])
                num_3d = len([k for k in data.keys() if k != '2D_keypoint'])
                has_annotation = (num_2d > 0) or (num_3d > 0)

                if has_annotation and first_annotated_frame is None:
                    first_annotated_frame = frame_idx

                dof = 3 * num_3d + 2 * num_2d
                if first_annotated_frame is not None and frame_idx >= first_annotated_frame and dof < 6:
                    invalid_frames.append((frame_idx, dof, num_3d, num_2d))

                merged[f"{frame_idx:05d}"] = data

        if first_annotated_frame is None:
            return jsonify({'error': 'No 2D or 3D annotations found in any frame; cannot save merged file'}), 400

        if invalid_frames:
            msg_lines = ["Frames with insufficient DoF (need >= 6):"]
            for frame_idx, dof, n3, n2 in invalid_frames[:10]:
                msg_lines.append(f"Frame {frame_idx}: DoF={dof} (3D={n3}x3, 2D={n2}x2)")
            if len(invalid_frames) > 10:
                msg_lines.append(f"... and {len(invalid_frames) - 10} more frames")
            return jsonify({'error': '\n'.join(msg_lines)}), 400

        # Attach global metadata: object scale and static flag
        if SCENE_DATA is not None:
            try:
                object_scale = float(getattr(SCENE_DATA, 'object_scale_factor', 1.0))
            except Exception:
                object_scale = 1.0
        else:
            object_scale = 1.0

        is_static_object = bool(payload.get('is_static_object', False))

        merged['object_scale'] = object_scale
        merged['is_static_object'] = is_static_object
        merged['start_frame_index'] = first_annotated_frame

        out_path = os.path.join(video_dir, 'kp_record_merged.json')
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(merged, f, indent=2, ensure_ascii=False)

        # 标注合并成功后，将对应视频的 annotation_progress 置为 3.0
        _update_hoi_progress_for_video_dir(video_dir, finished=True)

        return jsonify({
            'status': 'success',
            'path': out_path,
            'first_annotated_frame': first_annotated_frame,
            'last_frame_index': max_frame
        })

    except Exception as e:
        print(f"Error merging kp_record files: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Failed to merge kp_record files: {e}'}), 500

@app.route('/api/run_optimization', methods=['POST'])
def run_optimization():
    global SCENE_DATA, CAP
    
    if SCENE_DATA is None or not SCENE_DATA.loaded:
        return jsonify({'status': 'error', 'message': 'Scene data not loaded'})
        
    data = request.get_json(silent=True) or {}
    try:
        current_frame = int(data.get('frame_idx', 0))
    except:
        current_frame = 0
    
    # 1. Construct path to kp_record_merged.json
    video_dir = SCENE_DATA.video_dir
    kp_record_path = os.path.join(video_dir, 'kp_record_merged.json')
    
    if not os.path.exists(kp_record_path):
        return jsonify({'status': 'error', 'message': 'No merged annotations found. Please save annotations first.'})

    # Read merged json to get start_frame
    with open(kp_record_path, 'r') as f:
        merged = json.load(f)
    
    try:
        start_frame = merged.get("start_frame_index", 0)
    except:
        start_frame = 0
        
    # 2. Prepare arguments
    part_kp_path = os.path.join(app.root_path, 'solver', 'data', 'part_kp.json')
    if not os.path.exists(part_kp_path):
         return jsonify({'status': 'error', 'message': 'part_kp.json not found'})

    with open(part_kp_path, 'r') as f:
        human_part = json.load(f)

    # Prepare K
    width = int(CAP.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(CAP.get(cv2.CAP_PROP_FRAME_HEIGHT))
    focal_length = max(width, height) 
    cx = width / 2
    cy = height / 2
    K = np.array([
        [focal_length, 0, cx],
        [0, focal_length, cy],
        [0, 0, 1]
    ], dtype=np.float32)
    
    body_params = SCENE_DATA.motion_data['smpl_params_incam']
    
    # Prepare sampled meshes
    # Use SCENE_DATA.obj_mesh_org which is already simplified
    # We use the cached world meshes directly for optimization to support iterative refinement
    SCENE_DATA.update_world_meshes()
    sampled_orgs = SCENE_DATA.obj_orgs_world

    # Prepare original meshes (high res) - REMOVED as per user request
    # We only use sampled_orgs (world space meshes) now.


    # Since we are passing World Space meshes, we must set centers_depth to ZERO
    # to prevent double-transformation in the solver
    centers_depth = np.zeros((SCENE_DATA.total_frames, 3))

    try:
        if kp_use_new is None:
             return jsonify({'status': 'error', 'message': 'kp_use_new module not loaded'})

        # Debug: Print mesh sizes
        print(f"DEBUG: SCENE_DATA.obj_orgs[0] vertices: {len(SCENE_DATA.obj_orgs[0].vertices)}")
        print(f"DEBUG: sampled_orgs[0] vertices: {len(sampled_orgs[0].vertices)}")
        
        # Determine which mesh to use for constraints
        # We use sampled_orgs (world space meshes) for constraints

        
        # Run optimization
        new_body_params, new_icp_transforms = kp_use_new(
            output=None,
            hand_poses=SCENE_DATA.hand_poses,
            body_poses=body_params,
            global_body_poses=body_params,
            sampled_orgs=sampled_orgs,
            # centers_depth=centers_depth,
            human_part=human_part,
            K=torch.from_numpy(K),
            start_frame=start_frame,
            end_frame=SCENE_DATA.total_frames,
            video_dir=video_dir,
            is_static_object=False,
            kp_record_path=kp_record_path
        )
        
        # Apply transforms to object meshes in SCENE_DATA
        for i, transform in enumerate(new_icp_transforms):
            frame_idx = start_frame + i
            if frame_idx < len(SCENE_DATA.obj_orgs):
                mat_inc = transform.cpu().numpy() if hasattr(transform, 'cpu') else transform
                
                # Decompose incremental matrix into R_inc and t_inc
                R_inc = mat_inc[:3, :3]
                t_inc = mat_inc[:3, 3]
                
                # Update external transforms (t_finals, R_finals) by COMPOSING with existing transforms
                # T_new = T_inc * T_old
                # R_new = R_inc * R_old
                # t_new = R_inc * t_old + t_inc
                
                if SCENE_DATA.R_finals is not None and frame_idx < len(SCENE_DATA.R_finals):
                    R_old = SCENE_DATA.R_finals[frame_idx]
                    SCENE_DATA.R_finals[frame_idx] = R_inc @ R_old
                
                if SCENE_DATA.t_finals is not None and frame_idx < len(SCENE_DATA.t_finals):
                    t_old = SCENE_DATA.t_finals[frame_idx]
                    # Note: We need R_inc here as well for translation update
                    SCENE_DATA.t_finals[frame_idx] = R_inc @ t_old + t_inc

        # Update cached world meshes for visualization
        SCENE_DATA.update_world_meshes()

        return jsonify({'status': 'success'})
        
    except Exception as e:
        traceback.print_exc()
        return jsonify({'status': 'error', 'message': str(e)})

def load_video_frames():
    global VIDEO_FRAMES, VIDEO_FRAMES_ENCODED, CAP, VIDEO_FPS, VIDEO_TOTAL_FRAMES
    if CAP is None:
        return

    VIDEO_FPS = CAP.get(cv2.CAP_PROP_FPS)
    VIDEO_TOTAL_FRAMES = int(CAP.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Loading video frames into memory... FPS: {VIDEO_FPS}, Total frames: {VIDEO_TOTAL_FRAMES}")
    VIDEO_FRAMES = []
    VIDEO_FRAMES_ENCODED = []

    # JPEG编码参数：质量85，快速编码
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, 85]

    CAP.set(cv2.CAP_PROP_POS_FRAMES, 0)
    frame_count = 0

    while True:
        ret, frame = CAP.read()
        if not ret:
            break

        # 存储原始RGB帧（用于其他处理）
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        VIDEO_FRAMES.append(frame_rgb)

        # 预编码为JPEG用于快速响应
        _, buffer = cv2.imencode('.jpg', frame, encode_params)
        VIDEO_FRAMES_ENCODED.append(buffer.tobytes())

        frame_count += 1
        if frame_count % 50 == 0:
            print(f"Loaded and encoded {frame_count}/{VIDEO_TOTAL_FRAMES} frames...")

    print(f"Successfully loaded {len(VIDEO_FRAMES)} frames")
    print(f"Total encoded size: {sum(len(f) for f in VIDEO_FRAMES_ENCODED) / 1024 / 1024:.2f} MB")

def init_cotracker():
    global COTRACKER_MODEL
    if not COTRACKER_AVAILABLE:
        return
    
    checkpoint_path = os.path.join(os.path.dirname(__file__), 'co-tracker/checkpoints/scaled_online.pth')
    if os.path.exists(checkpoint_path):
        print(f"Loading CoTracker from {checkpoint_path}")
        COTRACKER_MODEL = CoTrackerOnlinePredictor(checkpoint=checkpoint_path)
        if torch.cuda.is_available():
            COTRACKER_MODEL = COTRACKER_MODEL.to('cuda')
        print("CoTracker loaded")
    else:
        print(f"CoTracker checkpoint not found at {checkpoint_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_dir', type=str, help='Path to the video directory')
    args = parser.parse_args()
    
    if args.video_dir:
        VIDEO_PATH = os.path.join(args.video_dir, "video.mp4")
        OBJ_PATH = os.path.join(args.video_dir, "obj_org.obj")
        
        if os.path.exists(VIDEO_PATH):
            CAP = cv2.VideoCapture(VIDEO_PATH)
            print(f"Video loaded: {VIDEO_PATH}")
            load_video_frames()
            init_cotracker()
            
            # Initialize Scene Data
            SCENE_DATA = SceneData(args.video_dir)
            SCENE_DATA.load()
        else:
            print(f"Warning: Video not found at {VIDEO_PATH}")
            
        # 复用通用的 session 加载逻辑，确保与 /api/hoi_start 行为一致
        _load_video_session(args.video_dir)
    else:
        print("No video_dir provided. Start with --video_dir to load data.")

    # 启动 Flask 之前初始化 HOI 待标注任务列表（不会修改 2.0，只清理历史 2.1）
    _init_hoi_tasks()

    app.run(debug=True, host=CONFIG['server']['host'], port=CONFIG['server']['port'], use_reloader=False)
