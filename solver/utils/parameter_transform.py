#!/usr/bin/env python3
import json
import numpy as np
import torch
from scipy.spatial.transform import Rotation
import os
import shutil


def compute_combined_transform(incam_params, global_params):
    incam_orient, incam_trans = incam_params
    global_orient, global_trans = global_params
    axis_angle = incam_orient.detach().cpu().numpy().copy()
    R_2ori = Rotation.from_rotvec(axis_angle).as_matrix()
    T_2ori = incam_trans.detach().cpu().numpy().squeeze().copy()
    axis_angle = global_orient.detach().cpu().numpy().copy()
    global_R = Rotation.from_rotvec(axis_angle).as_matrix()
    global_T = global_trans.detach().cpu().numpy().copy()
    R_combined = global_R @ R_2ori.T
    T_combined = global_R @ (-R_2ori.T @ T_2ori) + global_T
    return R_combined, T_combined


def apply_transform_to_smpl_params(global_orient, transl, incam_params, global_params):
    R_combined, T_combined = compute_combined_transform(incam_params, global_params)
    original_orient = global_orient.cpu().numpy().copy()
    original_R = Rotation.from_rotvec(original_orient).as_matrix()
    new_R = R_combined @ original_R
    new_orient = Rotation.from_matrix(new_R).as_rotvec()
    original_transl = transl.cpu().numpy().copy()
    if original_transl.ndim > 1:
        original_transl = original_transl.squeeze()
    new_transl = R_combined @ original_transl + T_combined
    return new_orient, new_transl
def transform_and_save_parameters(
    human_params_dict,
    org_params,
    camera_params,
    output_dir,
    original_object_path,
    user_scale=1.0,
):
    print("Transforming and saving parameters...")
    os.makedirs(output_dir, exist_ok=True)
    incam_params = camera_params["smpl_params_incam"]
    global_params = camera_params["smpl_params_global"]
    sorted_frames = sorted([int(k) for k in human_params_dict['body_pose'].keys()])
    transformed_human_params = {
        'body_pose': {},
        'betas': {},
        'global_orient': {},
        'global_orient_new': {},
        'transl': {},
        'transl_new': {},
        'left_hand_pose': {},
        'right_hand_pose': {},
    }

    print(f"Transforming human parameters for {len(sorted_frames)} frames...")

    def _get_cam_param(param_dict, frame_idx, fallback_idx):
        """Return camera param at absolute frame_idx, fallback to local index.

        camera_params often stores full-length lists indexed by absolute frame.
        When optimizing a subrange, using enumerate() index will misalign frames.
        """
        try:
            if isinstance(param_dict, (list, tuple)):
                if 0 <= frame_idx < len(param_dict):
                    return param_dict[frame_idx]
                return param_dict[fallback_idx]
            # dict-like
            if frame_idx in param_dict:
                return param_dict[frame_idx]
            if str(frame_idx) in param_dict:
                return param_dict[str(frame_idx)]
            return param_dict[fallback_idx]
        except Exception:
            return param_dict[fallback_idx]

    for local_id, frame_idx in enumerate(sorted_frames):
        frame_str = str(frame_idx)

        original_global_orient = torch.tensor(human_params_dict['global_orient'][frame_str], dtype=torch.float32)
        original_transl = torch.tensor(human_params_dict['transl'][frame_str], dtype=torch.float32)

        # Use absolute frame_idx whenever possible; fallback to local_id for older formats.
        incam_param = (
            _get_cam_param(incam_params['global_orient'], frame_idx, local_id),
            _get_cam_param(incam_params['transl'], frame_idx, local_id),
        )
        global_param = (
            _get_cam_param(global_params['global_orient'], frame_idx, local_id),
            _get_cam_param(global_params['transl'], frame_idx, local_id),
        )

        transformed_human_params['body_pose'][frame_str] = human_params_dict['body_pose'][frame_str]
        transformed_human_params['betas'][frame_str] = human_params_dict['betas'][frame_str]
        transformed_human_params['global_orient'][frame_str] = human_params_dict['global_orient'][frame_str]
        transformed_human_params['transl'][frame_str] = human_params_dict['transl'][frame_str]
        transformed_human_params['left_hand_pose'][frame_str] = human_params_dict['left_hand_pose'][frame_str]
        transformed_human_params['right_hand_pose'][frame_str] = human_params_dict['right_hand_pose'][frame_str]

    transformed_object_params = None
    transformed_object_path = None

    if 'poses' in org_params and org_params['poses'] and original_object_path and os.path.exists(original_object_path):
        print("Transforming object parameters and mesh...")

        transformed_object_params = {
            'R_total': {},
            'T_total': {},
        }

        for local_id, frame_idx in enumerate(sorted_frames):
            frame_str = str(frame_idx)

            if frame_str in org_params['poses'] and org_params['poses'][frame_str] is not None:
                R_final = np.array(org_params['poses'][frame_str], dtype=np.float32)
                t_final = np.array(org_params['centers'][frame_str], dtype=np.float32)

                incam_param = (
                    _get_cam_param(incam_params['global_orient'], frame_idx, local_id),
                    _get_cam_param(incam_params['transl'], frame_idx, local_id),
                )
                global_param = (
                    _get_cam_param(global_params['global_orient'], frame_idx, local_id),
                    _get_cam_param(global_params['transl'], frame_idx, local_id),
                )

                R_combined, T_combined = compute_combined_transform(incam_param, global_param)

                R_total = R_combined @ R_final
                T_total = R_combined @ t_final + T_combined

                transformed_object_params['R_total'][frame_str] = R_total.tolist()
                transformed_object_params['T_total'][frame_str] = T_total.tolist()
            else:
                transformed_object_params['R_total'][frame_str] = np.eye(3).tolist()
                transformed_object_params['T_total'][frame_str] = np.zeros(3).tolist()

        transformed_object_path = transform_and_save_object_mesh(original_object_path, output_dir)
        print("Transformed object parameters computed; mesh copied (no extra scaling).")

    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    saved_files = []
    transformed_data = {
        'metadata': {
            'description': 'Transformed parameters with all transforms applied',
            'total_frames': len(sorted_frames),
            'frame_indices': sorted_frames,
            'has_object_params': transformed_object_params is not None,
            # user_scale is kept only for backward compatibility; scale is baked into obj_init.obj.
        },
        'human_params': transformed_human_params
    }

    if transformed_object_params is not None:
        transformed_data['object_params'] = transformed_object_params

    transformed_params_path = os.path.join(output_dir, f'transformed_parameters_{timestamp}.json')
    with open(transformed_params_path, 'w') as f:
        json.dump(transformed_data, f, indent=2)
    saved_files.append(transformed_params_path)
    print(f"Saved transformed parameters: {transformed_params_path}")
    if transformed_object_path:
        saved_files.append(transformed_object_path)
    return saved_files
def transform_and_save_object_mesh(original_object_path, output_dir):
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_object_path = os.path.join(output_dir, f"transformed_object_{timestamp}.obj")

    try:
        shutil.copyfile(original_object_path, output_object_path)
        print(f"Object mesh copied to: {output_object_path}")
        return output_object_path
    except Exception as e:
        print(f"Error: Failed to copy mesh: {e}")
        return None
