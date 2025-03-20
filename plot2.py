import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
import os
from scipy.stats import gaussian_kde
from scipy import optimize
import gc
import pickle

def read_three_column_data(filename):
    """Read data from a three-column file format efficiently with memory constraints"""
    # Process file in smaller chunks to avoid memory issues
    chunks = pd.read_csv(filename, header=None, names=['traj_idx', 'time', 'value'], 
                     sep=r',\s*', chunksize=100000, engine='python')
       
    # Initialize empty arrays
    traj_indices, times, values = [], [], []
    
    # Process each chunk
    for chunk in chunks:
        traj_indices.append(chunk['traj_idx'].values)
        times.append(chunk['time'].values)
        values.append(chunk['value'].values)
    
    # Combine chunks
    return np.concatenate(traj_indices), np.concatenate(times), np.concatenate(values)

def compute_mle_pdf(data, threshold=0):
    """Compute PDF using Maximum Likelihood Estimation with memory optimizations"""
    filtered_data = data[data >= threshold]
    
    if len(filtered_data) == 0:
        return None, None, None, None, None, None
    
    # Subsample if too many points (saves memory with minimal precision loss)
    max_samples = 10000
    if len(filtered_data) > max_samples:
        indices = np.linspace(0, len(filtered_data)-1, max_samples, dtype=int)
        filtered_data = filtered_data[indices]
    
    # Use KDE with ML bandwidth selection for density estimation
    try:
        # Create kernel density estimate using Gaussian kernels with scott method (faster than silverman)
        kde = gaussian_kde(filtered_data, bw_method='scott')
        
        # Create x-range for smooth plotting
        x_range = np.linspace(min(filtered_data), max(filtered_data), 1000)
        
        # Evaluate the density at the grid points
        pdf_values = kde(x_range)
        
        # Calculate average value
        average_value = np.mean(filtered_data)
        
        # For histogram plotting compatibility, also compute a histogram
        # but use the KDE for the actual PDF estimation
        hist_values, bin_edges = np.histogram(filtered_data, bins=20, density=True)  # Reduced from 30
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        return x_range, pdf_values, average_value, bin_centers, hist_values, bin_edges
    
    except Exception as e:
        print(f"Error in KDE estimation: {e}")
        return None, None, None, None, None, None

def process_time_window(args):
    """Process a single time window for parallel execution"""
    traj_indices, times, values, window, threshold, data_type, output_dir = args
    mask = (times >= window[0]) & (times <= window[1])
    window_values = values[mask]
    
    if len(window_values) == 0:
        return window, None  # Ensure a tuple of two elements is returned
    
    # Save memory by computing results first
    result = compute_mle_pdf(window_values, threshold)
    if result[0] is None:  # If KDE fails, return window and None
        return window, None
    
    x_values, pdf_values, average_value, bin_centers, hist_values, bin_edges = result
    
    # Create a unique results ID for this window
    window_id = f"{data_type.replace(' ', '_')}_time_{window[0]:.3f}_{window[1]:.3f}"
    
    # Save results to disk to free memory
    with open(f"{output_dir}/temp_result_{window_id}.pkl", 'wb') as f:
        pickle.dump((window, x_values, pdf_values, average_value, bin_centers, hist_values, bin_edges), f)
    
    # Return a minimal result to indicate success
    return window, window_id

def plot_pdfs(traj_indices, times, values, time_windows, data_type, output_dir='plots', threshold=0):
    """Plot PDFs with MLE approach for a given dataset and time windows with memory optimizations"""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f"{output_dir}/temp", exist_ok=True)
    
    # Process time windows in smaller batches
    args_list = [(traj_indices, times, values, window, threshold, data_type, f"{output_dir}/temp") for window in time_windows]
    
    # Process in smaller batches to conserve memory
    batch_size = 5
    results = []
    
    for i in range(0, len(args_list), batch_size):
        print(f"Processing batch {i//batch_size + 1} of {len(args_list)//batch_size + 1}")
        batch_args = args_list[i:i+batch_size]
        with ProcessPoolExecutor(max_workers=6) as executor:
            batch_results = list(executor.map(process_time_window, batch_args))
            results.extend(batch_results)
        # Force garbage collection
        gc.collect()
    
    # Process results and create plots
    for result in results:
        if len(result) != 2:  # Skip invalid results
            print(f"Skipping invalid result: {result}")
            continue
        
        window, window_id = result
        if window_id is None:
            print(f"No data in time window {window} for {data_type}")
            continue
            
        # Load data from disk
        try:
            with open(f"{output_dir}/temp/temp_result_{window_id}.pkl", 'rb') as f:
                window, x_values, pdf_values, average_value, bin_centers, hist_values, bin_edges = pickle.load(f)
                
            if x_values is None:
                continue
                
            plt.figure(figsize=(10, 6))
            
            # Plot histogram as reference
            plt.bar(bin_centers, hist_values, width=np.diff(bin_edges), 
                    alpha=0.3, color='skyblue', label="Histogram (Reference)")
            
            # Plot MLE-based PDF
            plt.plot(x_values, pdf_values, 'r-', linewidth=2, label="Maximum Likelihood PDF")
            
            # Plot average value
            plt.axvline(average_value, color='green', linestyle='--', linewidth=2, 
                        label=f"Average = {average_value:.4f}")
            
            plt.xlabel(data_type, fontsize=12)
            plt.ylabel("Density", fontsize=12)
            plt.yscale('log')
            plt.title(f"PDF of {data_type} (Time Window: {window[0]:.3f}-{window[1]:.3f})\nMaximum Likelihood Estimation, Excluding values < {threshold}", fontsize=14)
            plt.grid(True, alpha=0.3)
            plt.legend(fontsize=10)
            
            # Save high-resolution figure
            plt.tight_layout()
            plt.savefig(f"{output_dir}/pdf_mle_{data_type.replace(' ', '_')}_time_{window[0]:.3f}_{window[1]:.3f}.pdf", dpi=300)  # Reduced DPI
            plt.close()
            
            # Remove temporary file
            os.remove(f"{output_dir}/temp/temp_result_{window_id}.pkl")
            
        except Exception as e:
            print(f"Error processing result for window {window}: {e}")
    
    # Clean up remaining temporary files
    for filename in os.listdir(f"{output_dir}/temp"):
        if filename.startswith("temp_result_"):
            os.remove(f"{output_dir}/temp/{filename}")

def compute_statistics_for_windows(traj_indices, times, values, time_windows, threshold=0):
    """Compute statistics for all time windows with memory optimizations"""
    averages = []
    valid_windows = []
    
    # Process windows in batches to save memory
    batch_size = 10
    for i in range(0, len(time_windows), batch_size):
        batch_windows = time_windows[i:i+batch_size]
        batch_averages = []
        batch_valid_windows = []
        
        for window in batch_windows:
            mask = (times >= window[0]) & (times <= window[1])
            window_values = values[mask]
            
            if len(window_values) == 0:
                continue
                
            filtered_values = window_values[window_values >= threshold]
            
            if len(filtered_values) > 0:
                batch_averages.append(np.mean(filtered_values))
                batch_valid_windows.append(window)
        
        averages.extend(batch_averages)
        valid_windows.extend(batch_valid_windows)
        
        # Clear memory
        gc.collect()
    
    return np.array(averages), valid_windows

def plot_speed_limit(time_windows, length_averages, deviation_averages, length_squared_averages, delta_t, output_dir='plots', threshold=0):
    """Plot the speed limit comparison with optimized memory usage"""
    os.makedirs(output_dir, exist_ok=True)
    
    time_window_midpoints = [window[0] + delta_t / 2 for window in time_windows]
    
    # For large datasets, potentially reduce number of points plotted
    max_points = 50
    if len(time_window_midpoints) > max_points:
        indices = np.linspace(0, len(time_window_midpoints)-1, max_points, dtype=int)
        time_window_midpoints = [time_window_midpoints[i] for i in indices]
        length_averages = length_averages[indices]
        deviation_averages = deviation_averages[indices]
        length_squared_averages = length_squared_averages[indices]
    
    # Calculate ratio for speed limit
    average_ratio = length_squared_averages / (deviation_averages)
    average_ratio2 = length_averages**2 / (deviation_averages)
    
    plt.figure(figsize=(10, 8))
    
    # Plot all metrics
    plt.plot(time_window_midpoints, average_ratio, marker='o', markersize=6, linestyle='-', 
             color='g', linewidth=2, label=r'$\frac{\langle\ell^2[x(t)]\rangle}{2\langle j[x(t)]\rangle}$')
    plt.plot(time_window_midpoints, time_window_midpoints, 'k--', linewidth=2, label=r'$\tau$')
    #plt.plot(time_window_midpoints, length_squared_averages, 'b-', linewidth=2, label=r'$\langle\ell[x(t)]^2\rangle$')
    plt.plot(time_window_midpoints, length_averages, 'b--', linewidth=2, label=r'$\langle\ell[x(t)]\rangle$')
    plt.plot(time_window_midpoints, deviation_averages, 'r-', linewidth=2, label=r'$\langle j[x(t)]\rangle$')
    plt.plot(time_window_midpoints, average_ratio2, marker='x', markersize=6, linestyle='-', 
             color='b', linewidth=2, label=r'$\frac{\langle\ell[x(t)]\rangle^2}{2\langle j[x(t)]\rangle}$')

    plt.xscale('log')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    plt.xlabel("Time", fontsize=14)
    plt.ylabel("Value", fontsize=14)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/speed_limit_mle_100000_t2.pdf", dpi=300)  # Reduced DPI
    plt.close()

def main():
    # Create output directory
    output_dir = 'plots_mle'
    os.makedirs(output_dir, exist_ok=True)
    
    # Set threshold for filtering probabilities
    probability_threshold = 0
    
    # Define time windows for analysis
    delta_t = 0.002
    time_end = 2
    time_windows = [(t, t + delta_t) for t in np.arange(0.0, time_end, delta_t)]
    
    # For visualization, reduce the number of plots
    num_windows_to_plot = 5  # Can be adjusted
    selected_windows = time_windows[:num_windows_to_plot]
    
    # Process each data file separately to conserve memory
    print("Processing stochastic length data...")
    traj_indices_length, times_length, values_length = read_three_column_data("stochastic_length_100000.dat")
    plot_pdfs(traj_indices_length, times_length, values_length, selected_windows, "Stochastic Length", output_dir, probability_threshold)
    length_averages, valid_windows_length = compute_statistics_for_windows(
        traj_indices_length, times_length, values_length, time_windows, probability_threshold)
    # Clear memory
    del traj_indices_length, times_length, values_length
    gc.collect()
    
    print("Processing stochastic length squared data...")
    traj_indices_length_squared, times_length_squared, values_length_squared = read_three_column_data("stochastic_length_squared_100000.dat")
    plot_pdfs(traj_indices_length_squared, times_length_squared, values_length_squared, selected_windows, "Stochastic Length Squared", output_dir, probability_threshold)
    length_squared_averages, valid_windows_length_squared = compute_statistics_for_windows(
        traj_indices_length_squared, times_length_squared, values_length_squared, time_windows, probability_threshold)
    # Clear memory
    del traj_indices_length_squared, times_length_squared, values_length_squared
    gc.collect()
    
    print("Processing stochastic deviation data...")
    traj_indices_deviation, times_deviation, values_deviation = read_three_column_data("stochastic_deviation_100000.dat")
    plot_pdfs(traj_indices_deviation, times_deviation, values_deviation, selected_windows, "Stochastic Deviation", output_dir, probability_threshold)
    deviation_averages, valid_windows_deviation = compute_statistics_for_windows(
        traj_indices_deviation, times_deviation, values_deviation, time_windows, probability_threshold)
    # Clear memory
    del traj_indices_deviation, times_deviation, values_deviation
    gc.collect()
    
    # Find common valid windows
    common_windows = []
    for window in time_windows:
        if (window in valid_windows_length and 
            window in valid_windows_length_squared and 
            window in valid_windows_deviation):
            common_windows.append(window)
    
    # Extract corresponding averages for common windows
    indices_length = [valid_windows_length.index(window) for window in common_windows]
    indices_length_squared = [valid_windows_length_squared.index(window) for window in common_windows]
    indices_deviation = [valid_windows_deviation.index(window) for window in common_windows]
    
    common_length_averages = length_averages[indices_length]
    common_length_squared_averages = length_squared_averages[indices_length_squared]
    common_deviation_averages = deviation_averages[indices_deviation]
    
    print("Plotting speed limit comparison...")
    # Plot speed limit comparison
    plot_speed_limit(common_windows, common_length_averages, common_deviation_averages, 
                     common_length_squared_averages, delta_t, output_dir, probability_threshold)
    
    print(f"All plots saved to {output_dir} directory. Used maximum likelihood estimation with threshold value of {probability_threshold} for filtering.")

if __name__ == "__main__":
    main()
