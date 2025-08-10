from scipy.spatial import KDTree
import numpy as np

def get_face_centroids(mesh):
    centroids = []
    for triangle in mesh.triangles:
        v1 = mesh.vertices[triangle[0]]
        v2 = mesh.vertices[triangle[1]]
        v3 = mesh.vertices[triangle[2]]
        centroids.append((np.array(v1)+np.array(v2)+np.array(v3))/3.0)
    return centroids

def reverse_decimation_mapping(original_mesh, simplified_mesh, simplified_labels):
    original_centroids = get_face_centroids(original_mesh)
    simplified_mesh_centroids = get_face_centroids(simplified_mesh)

    kdtree = KDTree(simplified_mesh_centroids)

    # Map simplified vertices to the original vertices
    vertex_mapping = {}

    for i, vertex in enumerate(original_centroids):
        # Find closest original vertex for each simplified vertex
        closest_vertex_index = kdtree.query(vertex, k=1)[1]
        vertex_mapping[i] = int(closest_vertex_index)

    # Initialize segmentation for original vertices
    original_mesh_labels = [-1] * len(original_centroids)

    # Map each original vertex's label from the nearest simplified vertex
    for original_index, simplified_index in vertex_mapping.items():
        label = simplified_labels[simplified_index]  # Get label of nearest simplified vertex
        original_mesh_labels[original_index] = label

    return original_mesh_labels