import numpy as np 
from pathlib import Path 
import os
import open3d as o3d
from pathlib import Path 
import glob 
from scipy.sparse import csr_matrix 
from scipy.sparse.csgraph import connected_components

def extract_teeth(mesh, segments, extract_tooth):
    """
    Extract each segmented tooth from the mesh and return as dictionary of mesh objects
    Only the largest connected component for each tooth is returned
    
    Returns:
        dict: Dictionary where keys are segment_ids and values are Open3D mesh objects
    """
    # Convert mesh vertices and triangles to numpy arrays
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    
    # Get unique segment IDs (each ID represents a tooth)
    unique_segments = extract_tooth #np.unique(segments)
    
    # Dictionary to store extracted teeth
    extracted_teeth = {}
    
    # Process each segment (tooth)
    for segment_id in unique_segments:
        # Skip segment 0 if it represents background/non-tooth area
        if segment_id == 0:
            continue
            
        # Get indices of faces belonging to this tooth
        tooth_face_indices = np.where(segments == segment_id)[0]
        
        if len(tooth_face_indices) == 0:
            print(f"No faces found for segment {segment_id}, skipping")
            continue
            
        # Get the triangles for this tooth
        tooth_triangles = triangles[tooth_face_indices]
        
        # Find connected components
        # First, build an adjacency matrix between faces
        face_count = len(tooth_triangles)
        
        # Create a dictionary to track shared vertices between faces
        edge_dict = {}
        
        # For each face
        for i, tri in enumerate(tooth_triangles):
            # For each edge in the face
            for j in range(3):
                v1, v2 = sorted([tri[j], tri[(j+1)%3]])  # Get vertices of the edge
                edge = (v1, v2)
                
                if edge in edge_dict:
                    # This edge is shared with another face
                    neighbor = edge_dict[edge]
                    # Build adjacency - faces i and neighbor are connected
                    if neighbor != i:  # Avoid self-loops
                        # Create adjacency
                        edge_dict[(v1, v2)] = i  # Update edge owner
                else:
                    # First time we see this edge
                    edge_dict[edge] = i
        
        # Build adjacency matrix for faces
        adjacency = np.zeros((face_count, face_count), dtype=bool)
        
        # For each face, find triangles that share an edge
        for i, tri in enumerate(tooth_triangles):
            for j in range(3):
                v1, v2 = sorted([tri[j], tri[(j+1)%3]])
                edge = (v1, v2)
                neighbor = edge_dict.get(edge)
                if neighbor is not None and neighbor != i:
                    adjacency[i, neighbor] = True
                    adjacency[neighbor, i] = True
        
        # Convert to sparse matrix and find connected components
        graph = csr_matrix(adjacency)
        n_components, labels = connected_components(graph, directed=False)
        
        print(f"Found {n_components} connected components for tooth segment {segment_id}")
        
        if n_components > 1:
            # Find the largest component
            component_sizes = np.bincount(labels)
            largest_component = np.argmax(component_sizes)
            
            # Filter triangles to keep only the largest component
            component_mask = (labels == largest_component)
            tooth_triangles = tooth_triangles[component_mask]
            tooth_face_indices = tooth_face_indices[component_mask]
            
            print(f"Keeping largest component with {component_sizes[largest_component]} faces")
        
        # Get all vertex indices used by this tooth
        vertex_indices = np.unique(tooth_triangles.flatten())
        
        # Create a mapping from original vertex indices to new indices
        vertex_map = {old_idx: new_idx for new_idx, old_idx in enumerate(vertex_indices)}
        
        # Get the vertices for this tooth
        tooth_vertices = vertices[vertex_indices]
        
        # Remap triangle indices to new vertex indices
        remapped_triangles = np.array([
            [vertex_map[idx] for idx in triangle]
            for triangle in tooth_triangles
        ])
        
        # Create a new mesh object for this tooth
        tooth_mesh = o3d.geometry.TriangleMesh()
        tooth_mesh.vertices = o3d.utility.Vector3dVector(tooth_vertices)
        tooth_mesh.triangles = o3d.utility.Vector3iVector(remapped_triangles)
        
        # Store the tooth mesh in the dictionary
        extracted_teeth[segment_id] = tooth_mesh
        
        print(f"Extracted tooth segment {segment_id} with {len(remapped_triangles)} faces")
        
    print(f"Extracted {len(extracted_teeth)} teeth")
    return extracted_teeth