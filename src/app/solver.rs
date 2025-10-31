use anyhow::{anyhow, Result};
use ndarray::{s, Array2, ArrayView1, ArrayView2}; // <-- Added 's' for slicing
use elkai_rs::DistanceMatrix;

// Helper function
fn cosine_distance(a: ArrayView1<f32>, b: ArrayView1<f32>) -> f32 {
    let dot = a.dot(&b);
    let norm_a = a.dot(&a).sqrt();
    let norm_b = b.dot(&b).sqrt();
    if norm_a == 0.0 || norm_b == 0.0 { return 2.0; } // Max distance
    let similarity = (dot / (norm_a * norm_b)).clamp(-1.0, 1.0);
    1.0 - similarity
}

// Build matrix function
pub fn build_distance_matrix(features: ArrayView2<f32>) -> Result<Array2<f32>> {
    let n = features.shape()[0];
    let mut dist_matrix = Array2::zeros((n, n));
    for i in 0..n {
        dist_matrix[[i, i]] = 0.0;
        for j in (i + 1)..n {
            let row_i = features.row(i);
            let row_j = features.row(j);
            let dist = cosine_distance(row_i, row_j);
            dist_matrix[[i, j]] = dist;
            dist_matrix[[j, i]] = dist;
        }
    }
    Ok(dist_matrix)
}

// TSP Solver function
pub fn solve_tsp(dist_matrix: &Array2<f32>) -> Result<Vec<usize>> {
    let n = dist_matrix.shape()[0];
    if n < 2 {
        return Ok((0..n).collect());
    }

    // --- 1. Find Start/End Candidates (Heuristic for Orientation) ---
    // We find the most dissimilar pair. This is NOT to break the cycle,
    // but to orient the final path.
    let mut max_overall_dist = -1.0;
    let mut start_candidate = 0;
    let mut end_candidate = 0;
    for i in 0..n {
        for j in (i + 1)..n {
            let dist = dist_matrix[[i, j]];
            if dist > max_overall_dist {
                max_overall_dist = dist;
                start_candidate = i;
                end_candidate = j;
            }
        }
    }
    println!("Heuristic: Start/End candidates (most dissimilar): Frame {} and Frame {}", start_candidate, end_candidate);

    // --- 2. Create TSP Matrix with a "Dummy Node" ---
    // This converts the problem from finding the shortest *cycle* (TSP)
    // to finding the shortest *path* (Hamiltonian Path).
    let n_tsp = n + 1; // n real frames + 1 dummy node
    let dummy_node_index = n; // The dummy node is at index 'n'

    // Create an (n+1) x (n+1) matrix, initialized to zeros
    let mut tsp_matrix = Array2::zeros((n_tsp, n_tsp));

    // Copy the original n x n distance matrix into the top-left corner
    tsp_matrix.slice_mut(s![0..n, 0..n]).assign(dist_matrix);

    // The distance from the dummy node to all real nodes (and back) is 0.
    // This makes the 'break' in the cycle free.
    // This is already handled by Array2::zeros.

    // --- 3. Solve the (N+1)-city TSP Cycle ---
    let scaling_factor = 1_000_000.0;
    if dist_matrix.iter().any(|&x| x.is_nan() || x.is_infinite()) {
         return Err(anyhow!("Distance matrix contains NaN or Inf values"));
    }
    // Note: We scale the (n+1) x (n+1) matrix
    let int_dist_matrix = tsp_matrix.mapv(|v| (v.max(0.0) * scaling_factor).round() as i32);
    let matrix_vec: Vec<Vec<i32>> = int_dist_matrix.outer_iter().map(|row| row.to_vec()).collect();
    let cities = DistanceMatrix::new(matrix_vec);
    let permutation: Vec<usize> = cities.solve(10); // Returns (n+1)-length cycle

    if permutation.len() != n_tsp {
        return Err(anyhow!("TSP solver returned permutation with incorrect length: expected {}, got {}", n_tsp, permutation.len()));
    }

    // --- 4. Extract Path by Removing Dummy Node ---
    // The optimal cycle will include the segment [..., F_end, dummy_node, F_start, ...]
    // We find the dummy node's position to extract the path.
    let dummy_pos_in_cycle = permutation.iter().position(|&x| x == dummy_node_index)
        .ok_or_else(|| anyhow!("Dummy node not found in TSP cycle"))?;

    // Rotate the permutation so the dummy node is at the end
    let final_start_index = (dummy_pos_in_cycle + 1) % n_tsp;
    let mut ordered_path_with_dummy = Vec::with_capacity(n_tsp);
    ordered_path_with_dummy.extend_from_slice(&permutation[final_start_index..]);
    ordered_path_with_dummy.extend_from_slice(&permutation[..final_start_index]);

    // ordered_path_with_dummy is now [F_start, F_next, ..., F_end, dummy_node_index]
    ordered_path_with_dummy.pop(); // Remove the dummy node
    let mut ordered_path = ordered_path_with_dummy; // Now has length 'n'

    if ordered_path.len() != n {
         return Err(anyhow!("Path extraction failed, incorrect length: expected {}, got {}", n, ordered_path.len()));
    }

    // --- 5. Orient the Path using Heuristic ---
    // The path is either [F_start, ..., F_end] or [F_end, ..., F_start].
    // We use our heuristic candidates to decide which direction is correct.
    let path_start_node = ordered_path[0];
    let path_end_node = ordered_path[n - 1];

    // Cost of current orientation: (path_start -> start_cand) + (path_end -> end_cand)
    let cost_forward = dist_matrix[[path_start_node, start_candidate]] 
                     + dist_matrix[[path_end_node, end_candidate]];

    // Cost of reversed orientation: (path_start -> end_cand) + (path_end -> start_cand)
    let cost_reversed = dist_matrix[[path_start_node, end_candidate]] 
                      + dist_matrix[[path_end_node, start_candidate]];

    if cost_reversed < cost_forward {
        println!("Path starts with {} and ends with {}.", path_start_node, path_end_node);
        println!("Heuristic suggests reversal (Start: {}, End: {}). Reversing final path.", start_candidate, end_candidate);
        ordered_path.reverse();
    } else {
         println!("Path starts with {} and ends with {}. Orientation matches heuristic.", path_start_node, path_end_node);
    }

    Ok(ordered_path)
}