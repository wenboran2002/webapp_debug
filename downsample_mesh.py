import open3d as o3d
import os

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_PATH = os.path.join(BASE_DIR, 'asset', 'obj_org.obj')
OUTPUT_PATH = os.path.join(BASE_DIR, 'asset', 'obj_downsampled.obj')
TARGET_VERTICES = 5000

def downsample_mesh():
    print(f"Loading mesh from {INPUT_PATH}...")
    mesh = o3d.io.read_triangle_mesh(INPUT_PATH)
    
    if not mesh.has_vertices():
        print("Error: Failed to load mesh or mesh is empty.")
        return

    num_vertices = len(mesh.vertices)
    num_faces = len(mesh.triangles)
    print(f"Original mesh has {num_vertices} vertices and {num_faces} faces.")
    
    if num_vertices <= TARGET_VERTICES:
        print("Mesh already has fewer vertices than target. Copying original file...")
        o3d.io.write_triangle_mesh(OUTPUT_PATH, mesh)
        return

    # Calculate target faces
    # V - 0.5F = 2 => F = 2(V - 2).
    target_faces = int(TARGET_VERTICES * 2)
    
    print(f"Simplifying mesh to approximately {target_faces} faces (to get ~{TARGET_VERTICES} vertices)...")
    
    # Open3D simplification
    mesh_simplified = mesh.simplify_quadric_decimation(target_number_of_triangles=target_faces)
    
    print(f"Simplified mesh has {len(mesh_simplified.vertices)} vertices and {len(mesh_simplified.triangles)} faces.")
    
    print(f"Saving simplified mesh to {OUTPUT_PATH}...")
    o3d.io.write_triangle_mesh(OUTPUT_PATH, mesh_simplified)
    print("Done.")

if __name__ == "__main__":
    downsample_mesh()
