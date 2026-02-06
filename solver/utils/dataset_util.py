import json
import os
from typing import Dict, List, Any, Optional

def load_upload_records(records_path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(records_path):
        print(f"upload_records.json not found: {records_path}")
        return []
    
    try:
        with open(records_path, 'r', encoding='utf-8') as f:
            records = json.load(f)
        return records if isinstance(records, list) else []
    except Exception as e:
        print(f"Failed to read upload_records.json: {e}")
        return []

def get_records_by_annotation_progress(annotation_progress: int, 
                                     records_path: str = "./upload_records.json") -> List[Dict[str, Any]]:
    records = load_upload_records(records_path)
    filtered_records = []
    for record in records:
        json_name = record.get("session_folder").split('/')[-1]
        json_path = os.path.join(record.get("session_folder"), f"{json_name}.json")
        if not os.path.exists(json_path):
            continue
        with open(json_path, 'r', encoding='utf-8') as f:
            merged = json.load(f)
        if len(merged) < 5 and isinstance(record, dict) and record.get("annotation_progress") == annotation_progress:
            continue
        if isinstance(record, dict) and record.get("annotation_progress") == annotation_progress:
            filtered_records.append(record)
    return filtered_records

def update_record_progress_by_session_folder(
    records: List[Dict[str, Any]], session_folder: str, new_progress: float
) -> None:
    """Update annotation_progress in-place for the record with matching session_folder."""
    for rec in records:
        if isinstance(rec, dict) and str(rec.get("session_folder", "")) == str(session_folder):
            rec["annotation_progress"] = new_progress
            return


def write_upload_records(records: List[Dict[str, Any]], records_path: str) -> None:
    """Write records list to upload_records.json (atomic write via .tmp)."""
    tmp_path = records_path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    os.replace(tmp_path, records_path)


def update_record_annotation_progress(record_id: str, new_progress: int, 
                                    records_path: str = "./upload_records.json") -> bool:
    if not os.path.exists(records_path):
        print(f"upload_records.json not found: {records_path}")
        return False
    
    try:
        with open(records_path, 'r', encoding='utf-8') as f:
            records = json.load(f)
        
        updated = False
        for record in records:
            if isinstance(record, dict) and record.get("id") == record_id:
                record["annotation_progress"] = new_progress
                updated = True
                break
        
        if not updated:
            print(f"Record with ID {record_id} not found")
            return False
        with open(records_path, 'w', encoding='utf-8') as f:
            json.dump(records, f, indent=2, ensure_ascii=False)
        
        return True
    except Exception as e:
        print(f"Failed to update annotation_progress: {e}")
        return False

def get_static_flag_from_merged(merged_path: str) -> bool:
    if not os.path.exists(merged_path):
        print(f"kp_record_merged.json not found: {merged_path}")
        return False
    try:
        with open(merged_path, 'r', encoding='utf-8') as f:
            merged = json.load(f)
        return merged.get("is_static_object", False)
    except Exception as e:
        print(f"Failed to read static flag: {e}")
        return False

def find_last_annotated_frame(kp_dir: str) -> int:
    """Return the max frame index from per-frame JSON files (XXXXX.json), or -1."""
    if not os.path.isdir(kp_dir):
        return -1
    frames = []
    for fname in os.listdir(kp_dir):
        if fname.endswith(".json") and len(fname) >= 5 and fname[:5].isdigit():
            frames.append(int(fname[:5]))
    return max(frames) if frames else -1


def validate_dof(kp_dir: str, start_frame: int, end_frame: int) -> bool:
    """Check each frame has enough DoF (>=6) from per-frame JSON files. Returns False if any frame fails."""
    invalid = []
    for frame_idx in range(start_frame, end_frame + 1):
        fpath = os.path.join(kp_dir, f"{frame_idx:05d}.json")
        if not os.path.exists(fpath):
            invalid.append((frame_idx, 0, 0, 0))
            continue
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            invalid.append((frame_idx, 0, 0, 0))
            continue
        num_2d = len(data.get("2D_keypoint", []) or [])
        num_3d = len([k for k in data.keys() if k != "2D_keypoint"])
        dof = 3 * num_3d + 2 * num_2d
        if dof < 6:
            invalid.append((frame_idx, dof, num_3d, num_2d))
    if invalid:
        print("Not enough DoF (>=6).")
        for frame_idx, dof, n3, n2 in invalid[:20]:
            print(f"  Frame {frame_idx}: DoF={dof} (3D={n3}x3, 2D={n2}x2)")
        if len(invalid) > 20:
            print(f"... and {len(invalid) - 20} more frames")
        return False
    return True


def validate_dof_from_merged(merged: dict, start_frame: int, end_frame: int) -> bool:
    """Check each frame in merged has enough DoF (>=6). Returns False if any frame fails."""
    invalid = []
    for frame_idx in range(start_frame, end_frame + 1):
        key = f"{frame_idx:05d}"
        data = merged.get(key)
        if not isinstance(data, dict):
            invalid.append((frame_idx, 0, 0, 0))
            continue
        num_2d = len(data.get("2D_keypoint", []) or [])
        num_3d = len([k for k in data.keys() if k != "2D_keypoint"])
        dof = 3 * num_3d + 2 * num_2d
        if dof < 6:
            invalid.append((frame_idx, dof, num_3d, num_2d))
    if invalid:
        print("Not enough DoF (>=6).")
        for frame_idx, dof, n3, n2 in invalid[:20]:
            print(f"  Frame {frame_idx}: DoF={dof} (3D={n3}x3, 2D={n2}x2)")
        if len(invalid) > 20:
            print(f"... and {len(invalid) - 20} more frames")
        return False
    return True


def validate_record_for_optimization(record: Dict[str, Any]) -> bool:
    if not isinstance(record, dict):
        return False
    required_fields = ["id", "session_folder", "object_category"]
    for field in required_fields:
        if not record.get(field):
            print(f"Record missing required field: {field}")
            return False
    session_folder = record.get("session_folder")
    if not session_folder or not os.path.exists(session_folder):
        print(f"Session folder does not exist: {session_folder}")
        return False
    required_files = [
        "video.mp4",
        "obj_org.obj",
        "obj_init.obj",
        "motion/result.pt",
    ]
    for file_path in required_files:
        full_path = os.path.join(str(session_folder), file_path)
        if not os.path.exists(full_path):
            print(f"Missing required file: {full_path}")
            return False
    return True