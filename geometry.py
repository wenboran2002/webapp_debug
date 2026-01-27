import torch
def rotation_matrix_to_angle_axis(rotation_matrix):
    """
    将 3x3 旋转矩阵转换为 Axis-Angle 向量 (Rodrigues)。
    Input: (B, 3, 3)
    Output: (B, 3)
    """
    quaternion = rotation_matrix_to_quaternion(rotation_matrix)
    return quaternion_to_angle_axis(quaternion)

def rotation_matrix_to_quaternion(rotation_matrix, eps=1e-6):
    """
    3x3 旋转矩阵 -> 四元数 (w, x, y, z)
    """
    x_dim = 1
    batch_dim = rotation_matrix.shape[0]
    
    m00 = rotation_matrix[:, 0, 0]
    m01 = rotation_matrix[:, 0, 1]
    m02 = rotation_matrix[:, 0, 2]
    m10 = rotation_matrix[:, 1, 0]
    m11 = rotation_matrix[:, 1, 1]
    m12 = rotation_matrix[:, 1, 2]
    m20 = rotation_matrix[:, 2, 0]
    m21 = rotation_matrix[:, 2, 1]
    m22 = rotation_matrix[:, 2, 2]

    trace = m00 + m11 + m22
    
    def safe_sqrt(x):
        return torch.sqrt(torch.clamp(x, min=0))

    q_abs = safe_sqrt(torch.stack([
        1.0 + trace,
        1.0 + m00 - m11 - m22,
        1.0 - m00 + m11 - m22,
        1.0 - m00 - m11 + m22
    ], dim=x_dim))

    # 选择最大分量以保证数值稳定性
    quat_by_rijk = torch.stack([
        torch.stack([q_abs[:, 0], m21 - m12, m02 - m20, m10 - m01], dim=-1),
        torch.stack([m21 - m12, q_abs[:, 1], m10 + m01, m02 + m20], dim=-1),
        torch.stack([m02 - m20, m10 + m01, q_abs[:, 2], m21 + m12], dim=-1),
        torch.stack([m10 - m01, m22 + m20, m21 + m12, q_abs[:, 3], ], dim=-1),
    ], dim=-2)

    flr = torch.argmax(q_abs, dim=-1)
    quat_candidates = quat_by_rijk / (2 * q_abs[..., None].max(dim=-1, keepdim=True)[0])
    
    # 按照 argmax 选出最稳的结果
    mask = torch.nn.functional.one_hot(flr, num_classes=4).to(rotation_matrix.dtype).unsqueeze(-1)
    quat = (quat_candidates * mask).sum(dim=-2)
    return quat

def quaternion_to_angle_axis(quaternion: torch.Tensor) -> torch.Tensor:
    """
    四元数 (w, x, y, z) -> Axis-Angle
    """
    # 归一化四元数
    q1 = quaternion[:, 1:]
    q0 = quaternion[:, 0]
    
    norm = torch.norm(q1, p=2, dim=1)
    
    # 防止除零
    epsilon = 1e-8
    mask = norm > epsilon
    
    angle_axis = torch.zeros_like(q1)
    
    angle = 2 * torch.atan2(norm, q0)
    
    # 有效值正常计算
    angle_axis[mask] = q1[mask] / norm[mask].unsqueeze(1) * angle[mask].unsqueeze(1)
    
    # 极小旋转近似处理 (Taylor expansion 或直接为0)
    # 这里直接保持0即可，因为初始化就是0
    
    return angle_axis