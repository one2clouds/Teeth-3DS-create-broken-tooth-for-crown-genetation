import os
import open3d as o3d
from pathlib import Path 
import glob 
from scipy.sparse import csr_matrix 
from scipy.sparse.csgraph import connected_components
import numpy as np
from scipy.spatial import Delaunay
import copy
from utils.jaw_classification import JawClassification
from utils.segmentation import Segmentation
from utils.bounding_box import get_bounding_boxes
from utils.request_response_for_jaw_label_bounding_box_extraction import get_response
from utility import extract_teeth


# remove non-connected smaller components from teeth_gums_mesh
def extract_teeth_and_gums(mesh, area_threshold=None, percentile_threshold=95):
    """
    Extract teeth and gums from a dental mesh by removing large triangles.
    Only keeps the largest connected component of the teeth and gums.
    
    Parameters:
    -----------
    mesh : o3d.geometry.TriangleMesh
        The original mesh containing teeth, gums, and base
    area_threshold : float, optional
        Maximum triangle area to keep (triangles larger than this will be removed)
        If None, percentile_threshold will be used instead
    percentile_threshold : float, optional
        Percentile value used to automatically determine area_threshold
        Only used if area_threshold is None
        
    Returns:
    --------
    teeth_gums_mesh : o3d.geometry.TriangleMesh
        Mesh containing only teeth and gums (small triangles), only largest connected component
    base_mesh : o3d.geometry.TriangleMesh
        Mesh containing only the base structure (large triangles)
    """
    # Get triangles and vertices
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    
    # Calculate area of each triangle
    triangle_areas = []
    for triangle in triangles:
        # Get vertices of the triangle
        v0 = vertices[triangle[0]]
        v1 = vertices[triangle[1]]
        v2 = vertices[triangle[2]]
        
        # Calculate two edges of the triangle
        edge1 = v1 - v0
        edge2 = v2 - v0
        
        # Calculate area using cross product
        cross_product = np.cross(edge1, edge2)
        area = 0.5 * np.linalg.norm(cross_product)
        triangle_areas.append(area)
    
    triangle_areas = np.array(triangle_areas)
    
    # Determine threshold if not provided
    if area_threshold is None:
        area_threshold = np.percentile(triangle_areas, percentile_threshold)
        print(f"Automatically determined area threshold: {area_threshold}")
    
    # Create mask for small triangles (teeth and gums)
    small_triangle_mask = triangle_areas < area_threshold
    
    # Get triangles for teeth and gums
    teeth_gums_triangles = triangles[small_triangle_mask]
    
    # Get triangles for base
    base_triangles = triangles[~small_triangle_mask]
    
    # Create new mesh for teeth and gums
    teeth_gums_mesh = o3d.geometry.TriangleMesh()
    teeth_gums_mesh.vertices = o3d.utility.Vector3dVector(vertices)
    teeth_gums_mesh.triangles = o3d.utility.Vector3iVector(teeth_gums_triangles)
    
    # Create new mesh for base
    base_mesh = o3d.geometry.TriangleMesh()
    base_mesh.vertices = o3d.utility.Vector3dVector(vertices)
    base_mesh.triangles = o3d.utility.Vector3iVector(base_triangles)
    
    # Remove unreferenced vertices
    teeth_gums_mesh.remove_unreferenced_vertices()
    base_mesh.remove_unreferenced_vertices()
    
    # Extract only the largest connected component from teeth_gums_mesh
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Error) as cm:
        triangle_clusters, cluster_n_triangles, cluster_area = teeth_gums_mesh.cluster_connected_triangles()
    
    triangle_clusters = np.asarray(triangle_clusters)
    cluster_n_triangles = np.asarray(cluster_n_triangles)
    cluster_area = np.asarray(cluster_area)
    
    if len(cluster_n_triangles) > 0:  # If there are clusters
        # Find the largest cluster
        largest_cluster_idx = np.argmax(cluster_n_triangles)
        
        # Create a mask for triangles in the largest cluster
        largest_cluster_triangles = np.asarray(teeth_gums_mesh.triangles)
        largest_cluster_mask = triangle_clusters == largest_cluster_idx
        
        # Create a new mesh with only the largest cluster
        largest_cluster_mesh = o3d.geometry.TriangleMesh()
        largest_cluster_mesh.vertices = teeth_gums_mesh.vertices
        largest_cluster_mesh.triangles = o3d.utility.Vector3iVector(
            largest_cluster_triangles[largest_cluster_mask])
        
        # Remove unreferenced vertices
        largest_cluster_mesh.remove_unreferenced_vertices()
        
        # Replace the teeth_gums_mesh with the largest cluster mesh
        teeth_gums_mesh = largest_cluster_mesh
    # Compute normals
    teeth_gums_mesh.compute_vertex_normals()
    base_mesh.compute_vertex_normals()
    return teeth_gums_mesh, base_mesh   


# this code extracts teeth and their gums. 
def extract_teeth_group_and_their_gums(mesh, segments, target_segment_ids, min_triangles=100, gum_extension_factor=0.5):
    """
    Extract teeth and their corresponding gums from the mesh based on multiple segment IDs.
    First removes small disconnected components from teeth selection, then extends to include gums.
    
    Parameters:
    -----------
    mesh : o3d.geometry.TriangleMesh
        The original mesh containing all teeth and gums
    segments : np.ndarray or list
        Array indicating which segment each triangle belongs to
    target_segment_ids : list
        List of segment IDs of the teeth to extract
    min_triangles : int, optional (default=100)
        Minimum number of triangles for a component to be kept
    gum_extension_factor : float, optional (default=0.5)
        Factor to determine how much of the gum to include below the teeth
        
    Returns:
    --------
    extracted_result : o3d.geometry.TriangleMesh
        Mesh containing the extracted teeth and corresponding gums
    remaining_mesh : o3d.geometry.TriangleMesh
        Mesh with the extracted parts removed
    result_segments : np.ndarray
        Segments information for the triangles in the result mesh
    """
    # Ensure segments is a numpy array
    segments = np.asarray(segments)
    
    # Get triangles and vertices of the original mesh
    triangles = np.asarray(mesh.triangles)
    vertices = np.asarray(mesh.vertices)
    
    # Create a mask for triangles in any of the target segments
    teeth_mask = np.zeros_like(segments, dtype=bool)
    for segment_id in target_segment_ids:
        teeth_mask = teeth_mask | (segments == segment_id)
    
    # Check if we found any triangles in the target segments
    if not np.any(teeth_mask):
        print(ValueError(f"No triangles found for segment IDs {target_segment_ids}"))
    
    # Get triangles for the teeth selection
    teeth_triangles = triangles[teeth_mask]
    
    # Create a new mesh for the teeth group
    teeth_mesh = o3d.geometry.TriangleMesh()
    
    # Find all unique vertices used in the teeth triangles
    unique_vertices = np.unique(teeth_triangles.flatten())
    
    # Create new vertices array containing only the used vertices
    new_vertices = vertices[unique_vertices]
    
    # Create mapping from old indices to new indices
    vertex_map = {old_idx: new_idx for new_idx, old_idx in enumerate(unique_vertices)}
    
    # Remap triangle indices
    new_triangles = np.array([[vertex_map[v] for v in triangle] for triangle in teeth_triangles])
    
    # Set the teeth mesh geometry
    teeth_mesh.vertices = o3d.utility.Vector3dVector(new_vertices)
    teeth_mesh.triangles = o3d.utility.Vector3iVector(new_triangles)
    
    # Store the original teeth mesh triangle indices in the target mask order
    original_teeth_triangle_indices = np.where(teeth_mask)[0]
    
    # Store the segment information for the teeth triangles
    teeth_segments = segments[teeth_mask]
    
    # STEP 1: CLEAN THE TEETH SELECTION BY REMOVING SMALL COMPONENTS
    # Cluster the mesh to identify connected components
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Error) as cm:
        cluster_indices = teeth_mesh.cluster_connected_triangles()
    
    # Get component sizes and indices
    component_indices = np.asarray(cluster_indices[0])  # triangle -> component ID
    component_sizes = np.asarray(cluster_indices[1])    # component size (in triangles)
    
    # Create a mask for triangles in components large enough to keep
    keep_triangle_mask = np.array([component_sizes[c_idx] >= min_triangles for c_idx in component_indices])
    
    if not np.any(keep_triangle_mask):
        print("Warning: All components are smaller than the minimum size. Keeping the largest component.")
        largest_component_idx = np.argmax(component_sizes)
        keep_triangle_mask = (component_indices == largest_component_idx)
    
    # Get the indices of the triangles to keep
    kept_triangle_indices = np.where(keep_triangle_mask)[0]
    
    # Create a new mesh with only the kept triangles
    cleaned_teeth_triangles = np.asarray(teeth_mesh.triangles)[kept_triangle_indices]
    
    # Create a cleaned teeth mesh
    cleaned_teeth_mesh = o3d.geometry.TriangleMesh()
    
    # Find unique vertices used in cleaned triangles
    all_vertex_indices = cleaned_teeth_triangles.flatten()
    unique_cleaned_vertices = np.unique(all_vertex_indices)
    
    # Create new vertices array
    cleaned_teeth_vertices = np.asarray(teeth_mesh.vertices)[unique_cleaned_vertices]
    
    # Create mapping from old to new indices
    cleaned_vertex_map = {old_idx: new_idx for new_idx, old_idx in enumerate(unique_cleaned_vertices)}
    
    # Remap triangle indices
    cleaned_teeth_triangles_remapped = np.array([[cleaned_vertex_map[v] for v in triangle] for triangle in cleaned_teeth_triangles])
    
    # Set the cleaned teeth mesh geometry
    cleaned_teeth_mesh.vertices = o3d.utility.Vector3dVector(cleaned_teeth_vertices)
    cleaned_teeth_mesh.triangles = o3d.utility.Vector3iVector(cleaned_teeth_triangles_remapped)
    
    # Update original triangles mask to reflect only the kept teeth triangles
    original_clean_teeth_indices = original_teeth_triangle_indices[kept_triangle_indices]
    clean_teeth_mask = np.zeros_like(segments, dtype=bool)
    clean_teeth_mask[original_clean_teeth_indices] = True
    
    # STEP 2: DETERMINE BOUNDING BOX OF CLEANED TEETH FOR GUM EXTENSION
    # Get vertices for the clean teeth triangles
    clean_teeth_vertex_indices = np.unique(triangles[clean_teeth_mask].flatten())
    clean_teeth_vertices = vertices[clean_teeth_vertex_indices]
    
    # Determine the bounding box of the cleaned teeth
    min_bound = np.min(clean_teeth_vertices, axis=0)
    max_bound = np.max(clean_teeth_vertices, axis=0)
    
    # For dental models, typically y-axis is the vertical dimension
    vertical_axis = 1  # y-axis (0 for x-axis, 2 for z-axis)
    
    # Original height of the teeth
    teeth_height = max_bound[vertical_axis] - min_bound[vertical_axis]
    
    # Extend the bounding box downward to include gums
    extended_min_bound = min_bound.copy()
    extended_min_bound[vertical_axis] -= gum_extension_factor * teeth_height
    
    # STEP 3: EXTEND TO INCLUDE GUMS BASED ON CLEAN TEETH BOUNDING BOX
    # Create a mask for all triangles within the extended bounding box (clean teeth + gums)
    extended_mask = np.zeros_like(segments, dtype=bool)
    
    # Include all clean teeth triangles
    extended_mask[clean_teeth_mask] = True
    
    # For each remaining triangle, check if any of its vertices are within the extended bounding box
    for i, triangle in enumerate(triangles):
        # Skip if already included in clean teeth
        if clean_teeth_mask[i]:
            continue
            
        triangle_vertices = vertices[triangle]
        
        # Check if any vertex of this triangle is within the extended bounding box
        for vertex in triangle_vertices:
            in_x_range = min_bound[0] <= vertex[0] <= max_bound[0]
            in_z_range = min_bound[2] <= vertex[2] <= max_bound[2]
            in_extended_y_range = extended_min_bound[1] <= vertex[1] <= max_bound[1]
            
            if in_x_range and in_extended_y_range and in_z_range:
                extended_mask[i] = True
                break
    
    # Get triangles for the extended region (clean teeth + gums)
    extended_triangles = triangles[extended_mask]
    
    # Create a new mesh for the extended region
    extended_mesh = o3d.geometry.TriangleMesh()
    
    # Find all unique vertices used in the extended triangles
    ext_unique_vertices = np.unique(extended_triangles.flatten())
    
    # Create new vertices array containing only the used vertices
    ext_new_vertices = vertices[ext_unique_vertices]
    
    # Create mapping from old indices to new indices
    ext_vertex_map = {old_idx: new_idx for new_idx, old_idx in enumerate(ext_unique_vertices)}
    
    # Remap triangle indices
    ext_new_triangles = np.array([[ext_vertex_map[v] for v in triangle] for triangle in extended_triangles])
    
    # Set the extended mesh geometry
    extended_mesh.vertices = o3d.utility.Vector3dVector(ext_new_vertices)
    extended_mesh.triangles = o3d.utility.Vector3iVector(ext_new_triangles)
    
    # Store the segment information for the extended triangles
    extended_segments = segments[extended_mask]
    
    # STEP 4: CLEAN THE EXTENDED MESH BY REMOVING SMALL COMPONENTS
    # But make sure we keep components connected to the teeth
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Error) as cm:
        ext_cluster_indices = extended_mesh.cluster_connected_triangles()
    
    # Get component sizes and indices
    ext_component_indices = np.asarray(ext_cluster_indices[0])
    ext_component_sizes = np.asarray(ext_cluster_indices[1])
    
    # Identify components large enough to keep
    ext_keep_triangle_mask = np.array([ext_component_sizes[c_idx] >= min_triangles for c_idx in ext_component_indices])
    
    if not np.any(ext_keep_triangle_mask):
        print("Warning: All extended components are smaller than the minimum size. Keeping the largest component.")
        largest_component_idx = np.argmax(ext_component_sizes)
        ext_keep_triangle_mask = (ext_component_indices == largest_component_idx)
    
    # Get the indices of the triangles to keep
    ext_kept_triangle_indices = np.where(ext_keep_triangle_mask)[0]
    
    # Create a new mesh with only the kept triangles
    final_triangles = np.asarray(extended_mesh.triangles)[ext_kept_triangle_indices]
    
    # Create a final result mesh
    final_result_mesh = o3d.geometry.TriangleMesh()
    
    # Find unique vertices used in final triangles
    final_vertex_indices = final_triangles.flatten()
    unique_final_vertices = np.unique(final_vertex_indices)
    
    # Create new vertices array
    final_vertices = np.asarray(extended_mesh.vertices)[unique_final_vertices]
    
    # Create mapping from old to new indices
    final_vertex_map = {old_idx: new_idx for new_idx, old_idx in enumerate(unique_final_vertices)}
    
    # Remap triangle indices
    final_triangles_remapped = np.array([[final_vertex_map[v] for v in triangle] for triangle in final_triangles])
    
    # Set the final result mesh geometry
    final_result_mesh.vertices = o3d.utility.Vector3dVector(final_vertices)
    final_result_mesh.triangles = o3d.utility.Vector3iVector(final_triangles_remapped)
    final_result_mesh.compute_vertex_normals()



    # keep_largest_component(final_result_mesh)

    # We use below approach, because sometimes, the abutment teeth might not be connected, and that can be removed as well, we determined the area of noise previously and setup min triangle as 5290 from that
    keep_large_components(final_result_mesh, min_triangles = 5290)
    
    # Update segments information to match the final result mesh
    result_segments = extended_segments[ext_kept_triangle_indices]
    
    # Create a new mesh for the remaining parts
    remaining_mesh = o3d.geometry.TriangleMesh()
    
    # Get triangles that are not in the extended mask or were removed in final cleaning
    final_mask = np.zeros_like(segments, dtype=bool)
    original_final_indices = np.where(extended_mask)[0][ext_kept_triangle_indices]
    final_mask[original_final_indices] = True
    
    remaining_triangles = triangles[~final_mask]
    
    # Check if we have any remaining triangles
    if remaining_triangles.size == 0:
        print("Warning: No remaining triangles after extraction")
        # Create an empty mesh with the original vertices
        remaining_mesh.vertices = o3d.utility.Vector3dVector(vertices)
        remaining_mesh.triangles = o3d.utility.Vector3iVector([])
    else:
        # Set the vertices and triangles for the remaining mesh
        remaining_mesh.vertices = o3d.utility.Vector3dVector(vertices)
        remaining_mesh.triangles = o3d.utility.Vector3iVector(remaining_triangles)
        
        # Remove unused vertices
        remaining_mesh.remove_unreferenced_vertices()
        
        # Compute normals
        remaining_mesh.compute_vertex_normals()
    
    return final_result_mesh, remaining_mesh, result_segments


def remove_small_components(mesh, min_triangles=None):
    """
    Remove small connected components from a triangle mesh.
    
    Args:
        mesh: Open3D triangle mesh
        min_triangles: Minimum number of triangles for a component to keep.
                      If None, keeps only the largest component.
    
    Returns:
        Cleaned mesh with small components removed
    """
    
    # Get connected components
    triangle_clusters, cluster_n_triangles, cluster_area = mesh.cluster_connected_triangles()
    triangle_clusters = np.asarray(triangle_clusters)
    cluster_n_triangles = np.asarray(cluster_n_triangles)
    
    print(f"Found {len(cluster_n_triangles)} connected components")
    print(f"Component sizes (triangles): {cluster_n_triangles}")
    
    if min_triangles is None:
        # Keep only the largest component
        largest_cluster_idx = cluster_n_triangles.argmax()
        triangles_to_remove = triangle_clusters != largest_cluster_idx
        print(f"Keeping largest component with {cluster_n_triangles[largest_cluster_idx]} triangles")
    else:
        # Keep components with at least min_triangles
        large_cluster_indices = np.where(cluster_n_triangles >= min_triangles)[0]
        triangles_to_remove = ~np.isin(triangle_clusters, large_cluster_indices)
        print(f"Keeping {len(large_cluster_indices)} components with >= {min_triangles} triangles")
    
    # Remove small components
    mesh.remove_triangles_by_mask(triangles_to_remove)
    mesh.remove_unreferenced_vertices()
    
    return mesh

# Method 1: Keep only the largest component
def keep_largest_component(mesh):
    print(f"Original mesh: {len(mesh.vertices)} vertices, {len(mesh.triangles)} triangles")
    
    cleaned_mesh = remove_small_components(mesh)
    
    print(f"Cleaned mesh: {len(cleaned_mesh.vertices)} vertices, {len(cleaned_mesh.triangles)} triangles")
    return cleaned_mesh


def keep_large_components(mesh, min_triangles=5290):
    print(f"Original mesh: {len(mesh.vertices)} vertices, {len(mesh.triangles)} triangles")
    
    cleaned_mesh = remove_small_components(mesh, min_triangles=min_triangles)
    
    print(f"Cleaned mesh: {len(cleaned_mesh.vertices)} vertices, {len(cleaned_mesh.triangles)} triangles")
    return cleaned_mesh


def extract_and_trim_tooth_mesh(mesh, segments, target_segment_id, keep_ratio=0.5):
    """
    Extract a specific tooth segment from the mesh and trim it to keep a specified portion of its height.
    
    Parameters:
    -----------
    mesh : o3d.geometry.TriangleMesh
        The original mesh containing all teeth
    segments : np.ndarray or list
        Array indicating which segment each triangle belongs to
    target_segment_id : int
        The segment ID of the tooth to extract
    keep_ratio : float
        The ratio of the tooth height to keep (default is 0.6 for 60%)
        
    Returns:
    --------
    trimmed_tooth : o3d.geometry.TriangleMesh
        Mesh containing the trimmed tooth
    remaining_mesh : o3d.geometry.TriangleMesh
        Mesh with the extracted tooth removed
    """
    # Ensure segments is a numpy array
    segments = np.asarray(segments)
    
    # Get triangles and vertices of the original mesh
    triangles = np.asarray(mesh.triangles)
    vertices = np.asarray(mesh.vertices)
    
    # Create a mask for triangles in the target segment
    target_mask = (segments == target_segment_id)
    
    # Check if we found any triangles in the target segment
    if not np.any(target_mask):
        raise ValueError(f"No triangles found for segment ID {target_segment_id}")
    
    # Get triangles for the target tooth
    tooth_triangles = triangles[target_mask]
    
    # Create a new mesh for the tooth
    tooth_mesh = o3d.geometry.TriangleMesh()
    
    # Create a mapping from old vertex indices to new ones
    # First, find all unique vertices used in the tooth triangles
    unique_vertices = np.unique(tooth_triangles.flatten())
    
    # Create new vertices array containing only the used vertices
    new_vertices = vertices[unique_vertices]
    
    # Determine the height range of the tooth
    z_min, z_max = new_vertices[:, 2].min(), new_vertices[:, 2].max()
    z_threshold = z_min + keep_ratio * (z_max - z_min)
    
    # Create a mask to keep only the specified portion
    keep_mask = new_vertices[:, 2] < z_threshold
    
    # Filter vertices and update the mapping
    filtered_vertices = new_vertices[keep_mask]
    vertex_map = {old_idx: new_idx for new_idx, old_idx in enumerate(unique_vertices[keep_mask])}
    
    # Remap triangle indices and filter out triangles with removed vertices
    new_triangles = []
    for triangle in tooth_triangles:
        if all(v in vertex_map for v in triangle):
            new_triangles.append([vertex_map[v] for v in triangle])
    
    # Set the tooth mesh geometry
    tooth_mesh.vertices = o3d.utility.Vector3dVector(filtered_vertices)
    tooth_mesh.triangles = o3d.utility.Vector3iVector(new_triangles)
    
    # Create a new mesh for the remaining teeth
    remaining_mesh = o3d.geometry.TriangleMesh()
    
    # Get triangles that are not in the target segment
    remaining_triangles = triangles[~target_mask]
    
    # Check if we have any remaining triangles
    if remaining_triangles.size == 0:
        print("Warning: No remaining triangles after extraction")
        # Create an empty mesh with the original vertices
        remaining_mesh.vertices = o3d.utility.Vector3dVector(vertices)
        remaining_mesh.triangles = o3d.utility.Vector3iVector([])
    else:
        # Set the vertices and triangles for the remaining mesh
        remaining_mesh.vertices = o3d.utility.Vector3dVector(vertices)
        remaining_mesh.triangles = o3d.utility.Vector3iVector(remaining_triangles)
        
        # Remove unused vertices
        remaining_mesh.remove_unreferenced_vertices()
    
    # Compute normals for both meshes
    tooth_mesh.compute_vertex_normals()
    if remaining_triangles.size > 0:
        remaining_mesh.compute_vertex_normals()
    
    return tooth_mesh, remaining_mesh


def fill_top_with_highest_point(mesh):
    """
    Create a triangular surface to cover the top of the cropped tooth using the highest point.
    
    Parameters:
    -----------
    mesh : o3d.geometry.TriangleMesh
        The original mesh with a cropped tooth
    
    Returns:
    --------
    filled_mesh : o3d.geometry.TriangleMesh
        Mesh with the top surface filled using the highest point
    """
    print("Filling the top of the cropped tooth using the highest point...")

    # Step 1: Extract mesh data
    points = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)

    # Step 2: Find boundary edges
    edges = set()
    boundary_edges = set()

    for triangle in triangles:
        e1 = tuple(sorted([triangle[0], triangle[1]]))
        e2 = tuple(sorted([triangle[1], triangle[2]]))
        e3 = tuple(sorted([triangle[0], triangle[2]]))

        for edge in [e1, e2, e3]:
            if edge in edges:
                boundary_edges.discard(edge)
            else:
                edges.add(edge)
                boundary_edges.add(edge)

    if not boundary_edges:
        print("No boundary detected.")
        return mesh

    # Step 3: Create a set of boundary vertices
    boundary_vertices = set()
    for edge in boundary_edges:
        boundary_vertices.add(edge[0])
        boundary_vertices.add(edge[1])

    boundary_vertices = list(boundary_vertices)
    boundary_points = points[boundary_vertices]

    # Step 4: Determine the highest point
    highest_point = boundary_points[np.argmax(boundary_points[:, 2])]

    # Step 5: Filter boundary vertices to only include those near the top
    # Define a threshold to consider a vertex as part of the top
    z_threshold = highest_point[2] - 0.1  # Adjust this threshold as needed

    top_boundary_vertices = [v for v in boundary_vertices if points[v][2] > z_threshold]
    top_boundary_points = points[top_boundary_vertices]

    if len(top_boundary_points) < 3:
        print("Not enough top boundary points detected.")
        return mesh

    # Step 6: Perform Delaunay triangulation on the top boundary points
    # Project points onto the XY plane for triangulation
    projected_points = top_boundary_points[:, :2]
    delaunay_triangles = Delaunay(projected_points).simplices

    # Step 7: Create the filling mesh
    filling_mesh = o3d.geometry.TriangleMesh()
    filling_mesh.vertices = o3d.utility.Vector3dVector(top_boundary_points)
    filling_mesh.triangles = o3d.utility.Vector3iVector(delaunay_triangles)

    # Step 8: Merge the meshes
    combined_mesh = mesh + filling_mesh

    # Clean up the combined mesh
    combined_mesh.remove_duplicated_vertices()
    combined_mesh.remove_duplicated_triangles()
    combined_mesh.remove_degenerate_triangles()
    combined_mesh.compute_vertex_normals()

    return combined_mesh

def merge_scaled_tooth_with_remaining(scaled_tooth, remaining_mesh):
    """
    Merge a scaled tooth mesh with the remaining teeth mesh
    
    Parameters:
    -----------
    scaled_tooth : o3d.geometry.TriangleMesh
        The scaled tooth mesh to merge
    remaining_mesh : o3d.geometry.TriangleMesh
        The mesh containing all remaining teeth
        
    Returns:
    --------
    merged_mesh : o3d.geometry.TriangleMesh
        A new mesh with the scaled tooth integrated into the remaining mesh
    """
    # Create a merged mesh
    merged_mesh = o3d.geometry.TriangleMesh()
    
    # Get vertices and triangles from both meshes
    vertices1 = np.asarray(remaining_mesh.vertices)
    triangles1 = np.asarray(remaining_mesh.triangles)
    vertices2 = np.asarray(scaled_tooth.vertices)
    triangles2 = np.asarray(scaled_tooth.triangles)
    
    # Calculate offset for the second mesh's triangle indices
    offset = len(vertices1)
    
    # Adjust triangle indices for the second mesh
    triangles2_adjusted = triangles2 + offset
    
    # Combine vertices and triangles
    combined_vertices = np.vstack((vertices1, vertices2))
    combined_triangles = np.vstack((triangles1, triangles2_adjusted))
    
    # Set the merged mesh geometry
    merged_mesh.vertices = o3d.utility.Vector3dVector(combined_vertices)
    merged_mesh.triangles = o3d.utility.Vector3iVector(combined_triangles)
    
    # Compute normals
    merged_mesh.compute_vertex_normals()
    return merged_mesh

def mesh_to_pcd(mesh, num_points = 8192):
    mesh = copy.deepcopy(mesh)
    mesh.compute_vertex_normals()
    o3d.utility.random.seed(12345)
    pcd = mesh.sample_points_uniformly(number_of_points=num_points)
    np_points = np.asarray(pcd.points)
    return pcd, np_points


def extract_context_teeth_surface(file_name, output_filename, three_teeth_segment_ids = [31, 32, 33], tooth_id_to_trim = 32):
    mesh = o3d.io.read_triangle_mesh(file_name)
    teeth_gums_mesh, _ = extract_teeth_and_gums(mesh, area_threshold=None, percentile_threshold=99)
    
    # saves_loss_of_vertices_while_not_saving
    teeth_gums_mesh.remove_degenerate_triangles()
    teeth_gums_mesh.remove_duplicated_triangles()
    teeth_gums_mesh.remove_duplicated_vertices()
    teeth_gums_mesh.remove_non_manifold_edges()

    response = get_response(teeth_gums_mesh)
    segment = response['labels']

    three_teeth_mesh, _, three_teeth_segments = extract_teeth_group_and_their_gums(teeth_gums_mesh, segment, three_teeth_segment_ids)
    trimmed_tooth, adjacent_teeth = extract_and_trim_tooth_mesh(three_teeth_mesh, three_teeth_segments, tooth_id_to_trim, keep_ratio=0.6)
    trimmed_filled_tooth = fill_top_with_highest_point(trimmed_tooth)
    context_tooth_surface = merge_scaled_tooth_with_remaining(trimmed_filled_tooth, adjacent_teeth)

    # os.makedirs("/home/shirshak/Teeth_3DS_data_preparation_for_reconstuction_and_generation/outputs/", exist_ok=True)
    # o3d.io.write_triangle_mesh("/home/shirshak/Teeth_3DS_data_preparation_for_reconstuction_and_generation/outputs/GT_tooth.ply", three_teeth_mesh)
    # o3d.io.write_triangle_mesh("/home/shirshak/Teeth_3DS_data_preparation_for_reconstuction_and_generation/outputs/final_tooth.ply", context_tooth_surface)

    # saves_loss_of_vertices_while_not_saving
    three_teeth_mesh.remove_degenerate_triangles()
    three_teeth_mesh.remove_duplicated_triangles()
    three_teeth_mesh.remove_duplicated_vertices()
    three_teeth_mesh.remove_non_manifold_edges()

    context_tooth_surface.remove_degenerate_triangles()
    context_tooth_surface.remove_duplicated_triangles()
    context_tooth_surface.remove_duplicated_vertices()
    context_tooth_surface.remove_non_manifold_edges()

    pcd_mesh_2_pcd_GT_tooth, npy_mesh_2_pcd_GT_tooth = mesh_to_pcd(three_teeth_mesh, num_points=4096) # 8192 points for Ground Truth
    #TODO 
    # 10240 for DMC PAPER
    pcd_mesh_2_pcd_context_tooth_surface, npy_mesh_2_pcd_context_tooth_surface = mesh_to_pcd(context_tooth_surface, num_points=4096) # 2048 points for 

    # np.save(f"outputs/DMC_GT/{output_filename}.npy", npy_mesh_2_pcd_GT_tooth)
    # np.save(f"outputs/DMC_input_context_surface/{output_filename}.npy", npy_mesh_2_pcd_context_tooth_surface)

    os.makedirs("outputs/DMC_input_context_surface/", exist_ok=True)
    o3d.io.write_point_cloud(os.path.join(f"outputs/DMC_input_context_surface/{output_filename}.pcd"), pcd_mesh_2_pcd_context_tooth_surface)

    os.makedirs("outputs/DMC_GT/", exist_ok=True)
    o3d.io.write_point_cloud(os.path.join(f"outputs/DMC_GT/{output_filename}.pcd"), pcd_mesh_2_pcd_GT_tooth)

    # EXTRACT INDIVIDUAL TOOTH FOR SHELL / CROWN 

    if "lower" in file_name:
        segment = response['labels']
        individual_extracted_teeth= extract_teeth(teeth_gums_mesh, np.array(segment), extract_tooth = [32])
        extract_tooth = 32
        pcd_individual_extracted_teeth, _ = mesh_to_pcd(individual_extracted_teeth[extract_tooth], num_points=1024)

        os.makedirs("outputs/", exist_ok=True)
        o3d.io.write_point_cloud(os.path.join(f"outputs/individual_{output_filename}_{extract_tooth}.pcd"), pcd_individual_extracted_teeth)


    # FOR DMC =====> EXTRACT INDIVIDUAL TEETH 
    # three_teeth_mesh, _, three_teeth_segments = extract_teeth_group_and_their_gums(mesh, segment, three_teeth_segment_ids)


# if __name__ == "__main__":
