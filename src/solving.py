import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import elkai  # <-- IMPORT THE NEW, SIMPLER SOLVER

# This function remains the same
def build_distance_matrix(features_list):
    """
    Creates an N x N matrix where (i, j) is the *dissimilarity*
    between frame i and frame j.
    """
    features_matrix = np.array(features_list)
    sim = cosine_similarity(features_matrix)
    dist_matrix = 1.0 - sim
    np.fill_diagonal(dist_matrix, 0)
    return dist_matrix


def solve_tsp(dist_matrix):
    """
    Solves the Traveling Salesperson Problem to find the optimal frame order
    using the high-performance elkai (LKH-3) solver.
    """
    
    # --- elkai Solver Integration ---
    # elkai also requires *integer* distances. We scale our
    # float distances (0.0 to 2.0) by a large factor to maintain precision.
    scaling_factor = 1_000_000
    int_dist_matrix = (dist_matrix * scaling_factor).astype(int)

    print("Running elkai (LKH-3) solver... (This will be *very* fast)")
    
    # elkai has a *direct* function for solving a matrix!
    # It returns the permutation (list of indices), starting from node 0.
    # Note: LKH node indices are 1-based, but elkai converts them to 0-based for Python.
    permutation = elkai.solve_int_matrix(int_dist_matrix)
    
    # --- Path-Cutting Heuristic (Same as before) ---
    # The solver finds the shortest *cycle* (A-B-C-D-A). We need a *path* (A-B-C-D).
    # We find the "longest link" (the most dissimilar pair, e.g., D-A)
    # in the cycle and "cut" it.
    
    num_frames = len(permutation)
    max_dist = -1.0
    cut_index = -1

    # Find the "longest link" in the cycle
    for i in range(num_frames):
        node_a = permutation[i]
        node_b = permutation[(i + 1) % num_frames] # Wrap around
        
        # We use the *original float matrix* for this check for best precision
        dist = dist_matrix[node_a, node_b]
        
        if dist > max_dist:
            max_dist = dist
            cut_index = i # This is the index of the "end" frame

    # Re-order the permutation to start *after* the cut
    start_index = (cut_index + 1) % num_frames
    ordered_path = permutation[start_index:] + permutation[:start_index]
    
    return ordered_path