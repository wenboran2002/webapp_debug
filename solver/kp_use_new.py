import json
import os
import numpy as np
from tqdm import tqdm
import torch

# from .optimizer_part import VideoBodyObjectOptimizer # 不需要视频级优化
from .hoi_solver import HOISolver
from .kp_use import (
    model,
    resource_path,
    apply_initial_transform_to_mesh,
    apply_initial_transform_to_points,
)
from copy import deepcopy

def kp_use_new(
    output,
    hand_poses,
    body_poses,
    global_body_poses,
    sampled_orgs,
    # centers_depth,
    human_part,
    K,
    start_frame,
    end_frame,
    video_dir,
    is_static_object=False,
    kp_record_path: str = None,
    save_debug_meshes: bool = False,
    debug_dir: str = None,
):
    if kp_record_path is None or not os.path.exists(kp_record_path):
        raise FileNotFoundError(f"kp_record_path not found: {kp_record_path}")
    with open(kp_record_path, "r", encoding="utf-8") as f:
        merged = json.load(f)
    body_params=body_poses
    global_body_params=global_body_poses
    seq_length = end_frame - start_frame
    best_frame = 0
    if is_static_object:
        max_count = -1
        for i in range(seq_length):
            frame_id = start_frame + i
            key = f"{frame_id:05d}"
            annotation = merged.get(key, {"2D_keypoint": []})
            num_2d = len(annotation.get("2D_keypoint", []))
            num_3d = 0
            for k in annotation.keys():
                if k in ("2D_keypoint", "multiview_2d_keypoints", "multiview_cam_params"):
                    continue
                num_3d += 1
            total = num_2d + num_3d
            if total > max_count:
                max_count = total
                best_frame = i

    object_points_idx = []
    body_points_idx = []
    pairs_2d = []
    object_points = []
    image_points = []
    body_kp_name = []
    
    # 修改模型路径，指向 plotly_4dhoi/asset/data/SMPLX_NEUTRAL.npz
    # 假设当前文件在 plotly_4dhoi/solver/ 下
    # 那么路径应该是 ../asset/data/SMPLX_NEUTRAL.npz
    # 使用 resource_path 处理
    model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../asset/data/SMPLX_NEUTRAL.npz'))
    if not os.path.exists(model_path):
        # Fallback to original logic if not found
        model_path = resource_path('video_optimizer/smpl_models/SMPLX_NEUTRAL.npz')
        
    hoi_solver = HOISolver(model_folder=model_path)

    if save_debug_meshes and not debug_dir:
        debug_dir = os.path.join(video_dir, "debug_meshes")
    if save_debug_meshes and debug_dir:
        os.makedirs(debug_dir, exist_ok=True)

    # IMPORTANT: sampled_orgs is indexed by ABSOLUTE frame (0..total_frames-1),
    # but this function iterates i in RELATIVE frame space (0..seq_length-1),
    # where abs_frame = start_frame + i. Always convert rel->abs when indexing
    # sampled_orgs, otherwise we will use the wrong mesh frame and introduce
    # systematic translation errors in visualization.
    def _abs_mesh_idx(rel_idx: int) -> int:
        abs_idx = int(start_frame) + int(rel_idx)
        if abs_idx < 0:
            return 0
        if abs_idx >= len(sampled_orgs):
            return len(sampled_orgs) - 1
        return abs_idx

    for i in tqdm(range(seq_length)):
        frame_id = start_frame + i
        key = f"{frame_id:05d}"
        annotation = merged.get(key, {"2D_keypoint": []})

        if annotation.get("2D_keypoint"):
            current_idx = best_frame if is_static_object else i
            point_indices = [p[0] for p in annotation["2D_keypoint"]]
            image_coords = [np.array(p[1]) for p in annotation["2D_keypoint"]]
            
            # Use sampled_orgs directly
            mesh_to_use = sampled_orgs[_abs_mesh_idx(current_idx)]
            
            object_verts = np.array(deepcopy(mesh_to_use.vertices))[point_indices]
            # transformed_verts = apply_initial_transform_to_points(
            #     object_verts, centers_depth[current_idx + start_frame]
            # )                              

            object_points.append(object_verts.astype(np.float32))
            image_points.append(np.array(image_coords, dtype=np.float32))
        else:
            object_points.append(np.array([]))
            image_points.append(np.array([]))
        object_idx = np.zeros((74, 2))
        for k, annot_index in annotation.items():
            if k in ("2D_keypoint", "multiview_2d_keypoints", "multiview_cam_params"):
                continue
            body_kp_name.append(k)
            human_part_index = list(human_part.keys()).index(k)
            object_idx[human_part_index] = [annot_index, 1]

        pairs_2d.append(annotation.get("2D_keypoint", []))
        body_idx = [v['index'] for v in human_part.values()]
        object_points_idx.append(object_idx)
        body_points_idx.append(body_idx)

    hoi_interval = 1
    if is_static_object:
        frames_to_optimize = [best_frame]
    else:
        frames_to_optimize = list(range(0, seq_length, hoi_interval))
        if frames_to_optimize[-1] != seq_length - 1:
            frames_to_optimize.append(seq_length - 1)

    optimized_results = {}
    icp_transform_matrix = []
    # 修改 joint_reflect.json 路径
    joint_mapping_path = os.path.join(os.path.dirname(__file__), 'data/joint_reflect.json')
    joint_mapping = json.load(open(joint_mapping_path))

    for i in frames_to_optimize:
                                                                                                               
        obj_src_idx = best_frame if is_static_object else i
        # obj_init = apply_initial_transform_to_mesh(
        #     sampled_orgs[obj_src_idx], centers_depth[obj_src_idx + start_frame]
        # )
        
        result = hoi_solver.solve_hoi(
            sampled_orgs[_abs_mesh_idx(obj_src_idx)],
            body_params,
            global_body_params,
            i,
            start_frame,
            end_frame,
            hand_poses,
            object_points_idx,
            body_points_idx,
            object_points,
            image_points,
            joint_mapping,
            K=K.cpu().numpy() if hasattr(K, "cpu") else K,
            is_multiview=False,
            save_meshes=save_debug_meshes,
            debug_dir=debug_dir,
            frame_id=(start_frame + i),
        )
        body_params['global_orient'][i + start_frame] = result['global_orient'].detach().cpu()
        body_params['body_pose'][i + start_frame] = result['body_pose'].detach().cpu()
        icp_transform_matrix.append(result['icp_transform_matrix'])

    if is_static_object:
        if len(icp_transform_matrix) > 0:
            icp_transform_matrix = [icp_transform_matrix[0] for _ in range(seq_length)]
        # first_frame_obj = obj_orgs[best_frame]
        # first_frame_sampled = sampled_orgs[best_frame]
        # for i in range(seq_length):
        #     obj_orgs[i] = first_frame_obj
        #     sampled_orgs[i] = first_frame_sampled

    # 不需要 VideoBodyObjectOptimizer
    # 直接返回优化后的 body_params 和 icp_transform_matrix
    
    # 构造返回结果
    # body_params 已经被原地修改了
    # icp_transform_matrix 是列表，包含了每一帧的变换矩阵 (4, 4)
    
    return body_params, icp_transform_matrix
