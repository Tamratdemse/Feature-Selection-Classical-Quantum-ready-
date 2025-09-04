import qci_client as qc
from quantum.multi_body_data import poly_coefficients, poly_indices
import numpy as np

# Initialize client and process the job
client = qc.QciClient()

data = []
for i in range(len(poly_coefficients)):
    data.append({
        "val": float(poly_coefficients[i]),
        "idx": poly_indices[i].tolist()
    })

poly_file = {
    "file_name": "test-polynomial",
    "file_config": {
        "polynomial": {
            "min_degree": 1,
            "max_degree": 2,
            "num_variables": 94,
            "data": data
        }
    }
}

file_id = client.upload_file(file=poly_file)["file_id"]

job_body = client.build_job_body(
    job_type="sample-hamiltonian", 
    polynomial_file_id=file_id,
    job_params={
        "device_type": "dirac-3",
        "sum_constraint": 1.584962500721156,
        "relaxation_schedule": 1,
        "num_samples": 1 # Get more samples for better results
    }
)

response = client.process_job(job_body=job_body)

# =========================================================
# Process the response and extract selected features
# =========================================================

def process_dirac_response(response, threshold=1e-4):
    """
    Process Dirac solver response and extract selected features based on threshold
    
    Args:
        response: The response from Dirac solver
        threshold: Minimum weight value to consider a feature as selected
        
    Returns:
        Dictionary with processed results
    """
    # Extract the solution (first sample)
    solution_vector = np.array(response['results']['solutions'][0])
    energy = response['results']['energies'][0]
    
    # Apply threshold to identify selected features
    selected_mask = solution_vector > threshold
    selected_indices = np.where(selected_mask)[0]
    selected_weights = solution_vector[selected_mask]
    
    # Calculate actual sum (may not exactly match constraint due to thresholding)
    actual_sum = np.sum(selected_weights)
    
    # Sort features by weight in descending order
    sorted_indices = selected_indices[np.argsort(selected_weights)[::-1]]
    sorted_weights = selected_weights[np.argsort(selected_weights)[::-1]]
    
    return {
        'energy': energy,
        'full_solution': solution_vector,
        'selected_indices': sorted_indices,
        'selected_weights': sorted_weights,
        'num_selected': len(selected_indices),
        'target_sum': 1.584962500721156,
        'actual_sum': actual_sum,
        'constraint_error': abs(actual_sum - 1.584962500721156)
    }

# Process the response
results = process_dirac_response(response)

# =========================================================
# Display the results
# =========================================================

print("=" * 60)
print("DIRAC SOLVER RESULTS - FEATURE SELECTION")
print("=" * 60)
print(f"Energy: {results['energy']:.6f}")
print(f"Number of selected features: {results['num_selected']}")
print(f"Target sum constraint: {results['target_sum']:.6f}")
print(f"Actual sum after thresholding: {results['actual_sum']:.6f}")
print(f"Constraint error: {results['constraint_error']:.6e}")
print("=" * 60)
print("TOP SELECTED FEATURES:")
print("Rank | Feature Index | Weight    | Importance")
print("-" * 45)

for i, (feature_idx, weight) in enumerate(zip(results['selected_indices'], results['selected_weights']), 1):
    print(f"{i:4d} | {feature_idx:12d} | {weight:8.6f} | {'*' * min(int(weight * 100), 20)}")

print("=" * 60)
print("SUMMARY:")
print(f"- {results['num_selected']} features selected out of 94")
print(f"- Sparsity ratio: {results['num_selected']/94:.1%}")
print(f"- Top feature: #{results['selected_indices'][0]} with weight {results['selected_weights'][0]:.6f}")
print("=" * 60)

# Save the results for later use
output = {
    'selected_features': results['selected_indices'].tolist(),
    'feature_weights': results['selected_weights'].tolist(),
    'full_solution': results['full_solution'].tolist(),
    'metadata': {
        'energy': results['energy'],
        'constraint_error': results['constraint_error'],
        'threshold_used': 1e-4
    }
}

print("✅ Feature selection completed successfully!")
print(f"✅ Selected {results['num_selected']} features with total weight {results['actual_sum']:.6f}")