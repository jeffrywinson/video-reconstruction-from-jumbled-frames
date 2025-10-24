import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from python_tsp.heuristics import solve_tsp_lin_kernighan

def build_distance_matrix(features_list):
    """
    Creates an N x N matrix where (i, j) is the *dissimilarity*
    between frame i and frame j.
    """
    # Convert list of vectors into a single N x D matrix
    features_matrix = np.array(features_list)
    
    # Calculate cosine similarity (values from -1 to 1)
    sim = cosine_similarity(features_matrix)
    
    # Convert to cosine distance (values from 0 to 2, 0=identical)
    # This will be our "cost" matrix for the TSP.
    dist_matrix = 1.0 - sim
    
    # Ensure the distance from a frame to itself is 0
    np.fill_diagonal(dist_matrix, 0)
    
    return dist_matrix

def solve_tsp(dist_matrix):
    """
    Solves the Traveling Salesperson Problem to find the optimal frame order.
    
    The LKH solver finds the shortest *cycle* (e.g., A-B-C-D-A).
    We need a *path* (e.g., A-B-C-D).
    
    Heuristic: The "end" and "start" frames (D -> A) are likely
    the most dissimilar pair in the solved cycle. We find this
    "longest link" and "cut" it to get a linear path.
    """
    # Solve for the optimal permutation (cycle)
    permutation, _ = solve_tsp_lin_kernighan(dist_matrix)
    
    num_frames = len(permutation)
    max_dist = -1.0
    cut_index = -1

    # Find the "longest link" in the cycle
    for i in range(num_frames):
        # Get the indices of two adjacent frames in the cycle
        node_a = permutation[i]
        node_b = permutation[(i + 1) % num_frames] # Wrap around
        
        dist = dist_matrix[node_a, node_b]
        
        if dist > max_dist:
            max_dist = dist
            cut_index = i # This is the index of the "end" frame

    # Re-order the permutation to create a linear path
    # The frame *after* the cut is the new start frame.
    start_index = (cut_index + 1) % num_frames
    
    # "Rotate" the list to start at the correct frame
    ordered_path = permutation[start_index:] + permutation[:start_index]
    
    return ordered_path