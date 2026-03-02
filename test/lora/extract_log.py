import re
from collections import defaultdict
import json

def extract_moe_saturation_data(log_file_path):
    """
    Parses the MoE profiling log to calculate the total unique experts fetched per execution batch.
    
    Returns:
        A dictionary mapping 'execution_batch_size' to a list of completed forward passes.
        Each forward pass is a list of exactly 48 integers representing the unique experts 
        activated in Layers 0 through 47.
    """
    # Key: execution_batch_size (total_tokens)
    # Value: List of forward passes (where each pass is a list of 48 integers)
    batch_data = defaultdict(list)
    
    # Regex to capture Layer ID, the contents of the activated_experts list, and the total_tokens
    log_pattern = re.compile(r'Layer (\d+):\s+activated_experts=\[(.*?)\],.*?total_tokens=(\d+)')
    
    current_pass = []
    current_batch_size = None
    last_processed_layer = -1
    
    with open(log_file_path, 'r') as file:
        for line in file:
            match = log_pattern.search(line)
            if not match:
                continue
                
            layer_id = int(match.group(1))
            experts_list_str = match.group(2).strip()
            total_tokens = int(match.group(3))
            
            # Skip the duplicate consecutive print statements
            if layer_id == last_processed_layer:
                continue
                
            # Calculate how many unique experts were activated
            num_unique_experts = len(experts_list_str.split(',')) if experts_list_str else 0
            
            # Detect the start of a new forward pass (Layer 0)
            if layer_id == 0:
                # If we were tracking a previous pass, save it
                if current_pass and current_batch_size is not None:
                    # Optional: Assert it has exactly 48 layers to ensure log integrity
                    # if len(current_pass) == 48: 
                    batch_data[current_batch_size].append(current_pass)
                
                # Reset for the new pass
                current_pass = []
                current_batch_size = total_tokens
                
            # Append the expert count for the current layer
            current_pass.append(num_unique_experts)
            last_processed_layer = layer_id

    # Don't forget to save the final pass in the file
    if current_pass and current_batch_size is not None:
         batch_data[current_batch_size].append(current_pass)

    return dict(batch_data)

# --- Analysis & Aggregation ---
def analyze_saturation(batch_data):
    """
    Calculates the cumulative sum of fetched experts per execution batch.
    """
    results = []
    for batch_size, passes in sorted(batch_data.items()):
        # Calculate the sum of experts across all 48 layers for each pass
        total_experts_per_pass = [sum(p) for p in passes]
        
        # Calculate the average across all passes with this batch size
        avg_total_experts = sum(total_experts_per_pass) / len(total_experts_per_pass)
        
        results.append({
            "execution_batch_size": batch_size,
            "avg_cumulative_experts_fetched": avg_total_experts,
            "sample_size": len(passes)
        })
    
    return results

if __name__ == "__main__":
    log_path = "/tmp/moe_profiling.log"  # Update with your actual path
    
    try:
        extracted_data = extract_moe_saturation_data(log_path)
        saturation_metrics = analyze_saturation(extracted_data)
        
        print(f"{'Execution Batch':<18} | {'Avg Cumulative Experts Fetched':<32} | {'Sample Passes'}")
        print("-" * 68)
        for metric in saturation_metrics:
            print(f"{metric['execution_batch_size']:<18} | {metric['avg_cumulative_experts_fetched']:<32.2f} | {metric['sample_size']}")
            
    except Exception as e:
        print(f"Error processing log: {e}")