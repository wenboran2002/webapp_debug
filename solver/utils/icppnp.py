import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R


def project_points(points_3d, K, T_ext=None):
    """Project 3D points to 2D with intrinsics K and optional extrinsics.

    This follows hoisolver_debug/test_alg.py:
      pts_h = [X,1]
      X_cam = T_ext @ pts_h
      uv = K @ (X_cam / z)

    Args:
        points_3d: (N,3) points in *world/object* coordinates.
        K: (3,3) camera intrinsics.
        T_ext: (4,4) extrinsic transform mapping points into camera coordinates.
               Default is identity (equivalent to "all-zeros" rvec/tvec).
    """
    points_3d = np.asarray(points_3d, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    if T_ext is None:
        T_ext = np.eye(4, dtype=np.float64)
    else:
        T_ext = np.asarray(T_ext, dtype=np.float64)
        if T_ext.shape != (4, 4):
            raise ValueError(f"T_ext must be 4x4, got shape {T_ext.shape}")

    pts_h = np.hstack((points_3d, np.ones((points_3d.shape[0], 1), dtype=np.float64)))
    p_cam = (T_ext @ pts_h.T).T[:, :3]

    # depth protection (match test_alg.py semantics)
    z = p_cam[:, 2:3]
    z = np.where(z < 0.1, 0.1, z)
    p_norm = p_cam / z

    uv_h = (K @ p_norm.T).T
    return uv_h[:, :2]

def residuals_weighted_priority(
    x,
    pts_3d_3d_src,
    pts_3d_3d_tgt,
    pts_3d_2d_src,
    pts_2d_tgt,
    K,
    T_ext=None,
    weight_3d=10.0,
    weight_2d=1.0,
):
    rvec = x[:3]
    tvec = x[3:]
    R_mat = R.from_rotvec(rvec).as_matrix()
    residual_all = []

    pts_3d_3d_trans = (R_mat @ pts_3d_3d_src.T).T + tvec
    res_3d = (pts_3d_3d_trans - pts_3d_3d_tgt).ravel()
    residual_all.append(weight_3d * res_3d)

    if K is not None:
        pts_3d_2d_world = (R_mat @ pts_3d_2d_src.T).T + tvec    
        proj_2d = project_points(pts_3d_2d_world, K, T_ext=T_ext)
        res_2d = (proj_2d - pts_2d_tgt).ravel()
        residual_all.append(weight_2d * res_2d)
    return np.hstack(residual_all)

def solve_weighted_priority(
    pts_3d_3d_src,
    pts_3d_3d_tgt,
    pts_3d_2d_src,
    pts_2d_tgt,
    K,
    T_ext=None,
    weight_3d=10.0,
    weight_2d=1.0,
):
    print("Starting optimization...", pts_3d_3d_src.shape, pts_3d_3d_tgt.shape, pts_3d_2d_src.shape, pts_2d_tgt.shape)
    x0 = np.zeros(6)
    residuals = residuals_weighted_priority(
        x0,
        pts_3d_3d_src,
        pts_3d_3d_tgt,
        pts_3d_2d_src,
        pts_2d_tgt,
        K,
        T_ext,
        weight_3d,
        weight_2d,
    )

    # SciPy LM requires m >= n (num residuals >= num variables). When constraints
    # are sparse (e.g. only a couple of 2D points), fall back to TRF so we can
    # return a best-effort solution (or a clear upstream validation error).
    method = 'lm'
    try:
        if residuals.size < x0.size:
            method = 'trf'
    except Exception:
        method = 'trf'

    res = least_squares(
        residuals_weighted_priority,
        x0,
        args=(pts_3d_3d_src, pts_3d_3d_tgt, pts_3d_2d_src, pts_2d_tgt, K, T_ext, weight_3d, weight_2d),
        method=method,
    )
    rvec_opt = res.x[:3]
    tvec_opt = res.x[3:]
    R_opt = R.from_rotvec(rvec_opt).as_matrix()
    return R_opt, tvec_opt


def visualize_projection_error(pts_3d_2d_src, pts_2d_tgt, R_opt, t_opt, K, image=None):
    def project_points(pts_3d, K):
        proj = pts_3d @ K.T
        proj = proj[:, :2] / proj[:, 2:3]
        return proj

    pts_3d_cam = (R_opt @ pts_3d_2d_src.T).T + t_opt
    pts_2d_proj = project_points(pts_3d_cam, K)

    plt.figure(figsize=(8, 6))
    if image is not None:
        plt.imshow(image)
    else:
        plt.gca().invert_yaxis()

    plt.scatter(pts_2d_tgt[:, 0], pts_2d_tgt[:, 1], c='r', label='Target 2D', s=50)
    plt.scatter(pts_2d_proj[:, 0], pts_2d_proj[:, 1], c='b', label='Projected 2D', s=50)

    for i in range(len(pts_2d_tgt)):
        plt.plot(
            [pts_2d_tgt[i, 0], pts_2d_proj[i, 0]],
            [pts_2d_tgt[i, 1], pts_2d_proj[i, 1]],
            'gray', linestyle='--', linewidth=1
        )

    plt.legend()
    plt.title("2D Projection Error")
    plt.xlabel("u (pixels)")
    plt.ylabel("v (pixels")
    plt.grid(True)
    plt.show()


def visualize_alignment(pts_src, pts_tgt, R_opt, t_opt):
    pts_src_trans = (R_opt @ pts_src.T).T + t_opt
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(pts_src[:, 0], pts_src[:, 1], pts_src[:, 2], c='g', label='Src 3D (before)', s=50)
    ax.scatter(pts_src_trans[:, 0], pts_src_trans[:, 1], pts_src_trans[:, 2], c='b', label='Src 3D (after)', s=50)
    ax.scatter(pts_tgt[:, 0], pts_tgt[:, 1], pts_tgt[:, 2], c='r', label='Target 3D', s=50)
    for i in range(len(pts_src)):
        ax.plot(
            [pts_src_trans[i, 0], pts_tgt[i, 0]],
            [pts_src_trans[i, 1], pts_tgt[i, 1]],
            [pts_src_trans[i, 2], pts_tgt[i, 2]],
            c='gray', linestyle='--', linewidth=1
        )
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()
    ax.set_title("3D Alignment Visualization")
    plt.tight_layout()
    plt.show()


def save_points_as_obj(points, filename):
    with open(filename, 'w') as f:
        for i, point in enumerate(points):
            f.write(f"v {point[0]} {point[1]} {point[2]}\n")
    print(f"Saved: {filename}")


if __name__ == "__main__":
    np.random.seed(42)
    K = np.array([[800, 0, 320], [0, 800, 240], [0, 0, 1]])
    n, m = 4, 3

    src_3d_3d = np.random.randn(n, 3) * 0.3 + [0, 0, 4]
    src_3d_2d = np.random.randn(m, 3) * 0.3 + [0, 0, 4]

    R_gt = R.from_euler("zyx", [5, -10, 15], degrees=True).as_matrix()
    t_gt = np.array([0.2, -0.1, 0.3])

    tgt_3d_3d = (R_gt @ src_3d_3d.T).T + t_gt + np.random.randn(n, 3) * 0.01
    tgt_2d = project_points((R_gt @ src_3d_2d.T).T + t_gt, K, T_ext=np.eye(4))
    tgt_2d += np.random.randn(m, 2) * 0.5

    R_est, t_est = solve_weighted_priority(src_3d_3d, tgt_3d_3d, src_3d_2d, tgt_2d, K, T_ext=np.eye(4))

    visualize_alignment(src_3d_3d, tgt_3d_3d, R_est, t_est)
    visualize_projection_error(src_3d_2d, tgt_2d, R_est, t_est, K)
