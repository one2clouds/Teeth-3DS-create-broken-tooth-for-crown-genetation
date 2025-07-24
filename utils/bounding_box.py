import numpy as np

def map_labels_to_vertices(mesh, labels):
    """
    Map labels to vertices based on the label assigned to each triangle.
    If a vertex is shared by multiple triangles, it will be added to the list of labels for each label.
    """
    # Create a dictionary to store the mapping {label: [vertices]}
    label_to_vertices = {}

    # Iterate over all triangles and assign labels to the vertices
    for triangle_idx, label in enumerate(labels):
        if label == 0:  # Skip label 0 (gingiva)
            continue
        triangle = mesh.triangles[triangle_idx]
        for vertex_idx in triangle:
            if label not in label_to_vertices:
                label_to_vertices[label] = []
            if vertex_idx not in label_to_vertices[label]:
                label_to_vertices[label].append(vertex_idx)

    return label_to_vertices


def get_largest_submesh(mesh, vertex_indices):
    """
    Extract the largest connected submesh from a list of vertex indices.
    """
    # Select the submesh corresponding to the vertex indices
    submesh = mesh.select_by_index(vertex_indices, cleanup=True)
    
    # Cluster connected components in the submesh
    triangle_clusters, _, _ = submesh.cluster_connected_triangles()
    
    # Count triangles in each connected component
    cluster_counts = np.bincount(triangle_clusters)
    
    # Get the largest component index
    largest_cluster_idx = np.argmax(cluster_counts)
    
    # Extract the triangle indices belonging to the largest cluster
    largest_cluster_triangle_indices = np.where(triangle_clusters == largest_cluster_idx)[0]
    
    # Select the vertices corresponding to the largest cluster of triangles
    largest_cluster_vertex_indices = set()
    for triangle_idx in largest_cluster_triangle_indices:
        triangle = submesh.triangles[triangle_idx]
        largest_cluster_vertex_indices.update(triangle)
    
    return submesh.select_by_index(list(largest_cluster_vertex_indices), cleanup=True)


def get_bounding_boxes(mesh, labels):
    """
    Compute the bounding box for each label in the mesh based on the triangle indices.
    """
    # Step 1: Map labels to vertices
    vertex_to_labels = map_labels_to_vertices(mesh, labels)

    bounding_box_dict = {}

    # Create the dict of labels with initialized max and min values
    list_of_unique_labels = list(set(labels))
    list_of_unique_labels.remove(0) # remove FDI 0 (i.e. gingiva)
    list_of_unique_labels.sort()
    for i in list_of_unique_labels:
        bounding_box_dict[i] = ([0,0,0], [0,0,0]) # in the form of tuple (min values, max values)
     

    # Step 2: Iterate over the labels and calculate the bounding box
    for label, vertices in vertex_to_labels.items():
        # Step 3: Get the largest submesh for the vertices corresponding to this label
        submesh = get_largest_submesh(mesh, vertices)
        
        # Get the vertices of the submesh
        submesh_vertices = np.asarray(submesh.vertices)
        
        # Initialize min/max values for bounding box
        min_values = [float('inf'), float('inf'), float('inf')]
        max_values = [float('-inf'), float('-inf'), float('-inf')]
        
        # Step 4: Compute bounding box for the vertices of this submesh
        for vertex in submesh_vertices:
            for k in range(3):  # Compare x, y, z values
                min_values[k] = min(min_values[k], vertex[k])
                max_values[k] = max(max_values[k], vertex[k])
        
        bounding_box_dict[label] = (min_values, max_values)
    
 
    return bounding_box_dict