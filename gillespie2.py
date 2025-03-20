import numpy as np
import matplotlib.pyplot as plt
from numba import jit
import multiprocessing as mp
from tqdm.auto import tqdm  # Auto-selects appropriate progress bar
import gc  # Garbage collection
import os

# Optimized rate functions using Numba
@jit(nopython=True)
def k_aa(t): return (np.arctan(t) + np.arctan(2*t))
@jit(nopython=True)
def k_ab(t): return (np.exp(-0.5))
@jit(nopython=True)
def k_ba(t): return np.arctan(2*t)
@jit(nopython=True)
def k_bb(t): return (np.exp(-0.5) + np.exp(-2.0)*np.arctan(2*t))
@jit(nopython=True)
def k_ac(t): return np.exp(-1.0)
@jit(nopython=True)
def k_ca(t): return np.arctan(t)
@jit(nopython=True)
def k_bc(t): return np.arctan(2*t)
@jit(nopython=True)
def k_cb(t): return np.exp(-2.0)*np.arctan(t)
@jit(nopython=True)
def k_cc(t): return (np.exp(-1.0) + np.arctan(2*t))

# Define rate functions in a numpy-friendly format for faster access
@jit(nopython=True)
def get_rate(from_state, to_state, t):
    if from_state == 0 and to_state == 0:
        return k_aa(t)
    elif from_state == 0 and to_state == 1:
        return k_ab(t)
    elif from_state == 1 and to_state == 0:
        return k_ba(t)
    elif from_state == 1 and to_state == 1:
        return k_bb(t)
    elif from_state == 0 and to_state == 2:
        return k_ac(t)
    elif from_state == 2 and to_state == 0:
        return k_ca(t)
    elif from_state == 1 and to_state == 2:
        return k_bc(t)
    elif from_state == 2 and to_state == 1:
        return k_cb(t)
    elif from_state == 2 and to_state == 2:
        return k_cc(t)
    else:
        return 0.0

# Simulation parameters
num_trajectories = 100000  # 10^6 trajectories
time_end = 0.2
num_time_points = 1000
time_points = np.linspace(0, time_end, num_time_points)
dt = time_points[1] - time_points[0]  # Time step
initial_probs = [0.9998, 0.0001, 0.0001]

@jit(nopython=True)
def gillespie_step(state, t, time_end):
    """Perform a single step of the Gillespie algorithm."""
    possible_states = [0, 1, 2]
    rates = np.zeros(3)
    
    for to_state in possible_states:
        rates[to_state] = get_rate(state, to_state, t)
    
    total_rate = np.sum(rates)
    
    if total_rate <= 1e-16:  # Avoid division by zero
        return state, time_end + 1  # Return a time beyond the end to terminate
    
    # Calculate time until next event
    tau = np.random.exponential(1.0 / total_rate)
    new_t = t + tau
    
    if new_t > time_end:
        return state, new_t
    
    # Choose the next state
    cumulative_rate = 0.0
    rand = np.random.uniform(0, total_rate)
    
    for to_state in possible_states:
        cumulative_rate += rates[to_state]
        if rand < cumulative_rate:
            return to_state, new_t
    
    # Fallback (should rarely happen)
    return state, new_t

def gillespie_simulation(seed=None):
    """Run a single Gillespie simulation with optional seed."""
    if seed is not None:
        np.random.seed(seed)
        
    t = 0.0
    state = np.random.choice([0, 1, 2], p=initial_probs)
    trajectory = [(t, state)]
    
    while t < time_end:
        state, t = gillespie_step(state, t, time_end)
        if t > time_end:
            break
        trajectory.append((t, state))
    
    return trajectory

def run_batch_simulations(batch_size, seed_offset=0):
    """Run a batch of simulations."""
    return [gillespie_simulation(seed=i+seed_offset) for i in range(batch_size)]

def calculate_probabilities_from_file(prob_file='state_probabilities.npy'):
    """Calculate or load state probabilities over time."""
    if os.path.exists(prob_file):
        # Load pre-calculated probabilities
        print(f"Loading pre-calculated probabilities from {prob_file}")
        probs = np.load(prob_file)
        return probs[0], probs[1], probs[2]
    
    # Use a smaller subset to calculate probabilities
    print("Calculating state probabilities using a subset of trajectories...")
    subset_size = min(num_trajectories, 10000)  # Use at most 10,000 trajectories
    
    # Reserve 2 cores for system operations, minimum 1 core
    num_cores = max(1, mp.cpu_count() - 2)
    batch_size = subset_size // num_cores
    remaining = subset_size % num_cores
    
    with mp.Pool(processes=num_cores) as pool:
        args = [(batch_size + (1 if i < remaining else 0), i * batch_size + min(i, remaining)) 
                for i in range(num_cores)]
        results = pool.starmap(run_batch_simulations, args)
    
    subset_trajectories = [traj for batch in results for traj in batch]
    print(f"Using {len(subset_trajectories)} trajectories to calculate probabilities")
    
    # Calculate probabilities
    P_A = np.zeros_like(time_points)
    P_B = np.zeros_like(time_points)
    P_C = np.zeros_like(time_points)
    
    for i, t in enumerate(time_points):
        states_at_t = []
        for traj in subset_trajectories:
            # Find the state at time t or the last state before t
            state = None
            for time, s in traj:
                if time <= t:
                    state = s
                else:
                    break
            if state is not None:
                states_at_t.append(state)
            elif traj:  # If trajectory is not empty
                states_at_t.append(traj[-1][1])
        
        if states_at_t:
            P_A[i] = states_at_t.count(0) / len(states_at_t)
            P_B[i] = states_at_t.count(1) / len(states_at_t)
            P_C[i] = states_at_t.count(2) / len(states_at_t)
    
    # Save for future use
    np.save(prob_file, np.array([P_A, P_B, P_C]))
    
    return P_A, P_B, P_C

def interpolate_trajectory_to_uniform_time_grid(trajectory, uniform_time_points):
    """Interpolate a trajectory to a uniform time grid."""
    if not trajectory:
        return []
    
    times, states = zip(*trajectory)
    interp_states = []
    
    for t in uniform_time_points:
        # Find the state at or before time t
        idx = np.searchsorted(times, t)
        if idx == 0:
            interp_states.append(states[0])  # Use first state for times before first event
        elif idx == len(times):
            interp_states.append(states[-1])  # Use last state for times after last event
        else:
            if times[idx] == t:
                interp_states.append(states[idx])  # Exact match
            else:
                interp_states.append(states[idx-1])  # Use previous state
    
    return list(zip(uniform_time_points, interp_states))

def process_trajectory_continuous(args):
    """Process a single trajectory to calculate stochastic quantities on a uniform time grid."""
    traj_idx, traj, P_A, P_B, P_C, uniform_time_points = args
    
    if not traj:
        return [], [], []
    
    # Interpolate trajectory to uniform time grid
    uniform_traj = interpolate_trajectory_to_uniform_time_grid(traj, uniform_time_points)
    
    times = [t for t, _ in uniform_traj]
    states = [s for _, s in uniform_traj]
    
    # Calculate entropy for each state
    S_t = []
    for i, t in enumerate(times):
        idx = min(np.searchsorted(time_points, t), len(time_points) - 1)
        state = states[i]
        prob = [P_A[idx], P_B[idx], P_C[idx]][state]
        entropy = -np.log(prob) if prob > 0 else 0
        S_t.append(entropy)
    
    stochastic_length = []
    stochastic_length_squared = []
    stochastic_deviation = []
    
    # Calculate entropy production for each time step
    entropy_production = np.zeros(len(times) - 1)
    for i in range(len(times) - 1):
        if states[i] != states[i+1]:  # State changed
            entropy_production[i] = (S_t[i+1] - S_t[i]) / (times[i+1] - times[i])
        # If state didn't change, entropy production remains 0
    
    # Calculate cumulative quantities
    stoch_length = 0
    stoch_length_squared = 0
    stoch_deviation = 0
    
    for i in range(len(times) - 1):
        delta_t = times[i+1] - times[i]
        delta_entropy = entropy_production[i] 
        
        stoch_length += np.abs(delta_entropy)*delta_t
        stochastic_length.append((traj_idx, times[i+1], stoch_length))
        
        stoch_length_squared += (np.abs(delta_entropy) * delta_t)**2
        stochastic_length_squared.append((traj_idx, times[i+1], stoch_length_squared))
        
        stoch_deviation += np.abs(delta_entropy)**2 * delta_t
        stochastic_deviation.append((traj_idx, times[i+1], stoch_deviation))
    
    return stochastic_length, stochastic_length_squared, stochastic_deviation

def save_data_chunk(data, filename, mode='a'):
    """Save a chunk of data to a file."""
    with open(filename, mode) as f:
        for traj_idx, t, val in data:
            f.write(f"{traj_idx}, {t}, {val}\n")

def simulate_and_process_chunk(chunk_idx, chunk_size, p_a, p_b, p_c):
    """Simulate and process a chunk of trajectories, saving results directly to disk."""
    start_idx = chunk_idx * chunk_size
    end_idx = min((chunk_idx + 1) * chunk_size, num_trajectories)
    current_chunk_size = end_idx - start_idx
    
    # Determine cores to use - leave 2 for system, minimum 1
    num_cores = max(1, mp.cpu_count() - 2)
    batch_size = current_chunk_size // num_cores
    remaining = current_chunk_size % num_cores
    
    # Generate chunk of trajectories
    print(f"Chunk {chunk_idx+1}: Generating {current_chunk_size} trajectories using {num_cores} cores")
    with mp.Pool(processes=num_cores) as pool:
        args = [(batch_size + (1 if i < remaining else 0), 
                start_idx + i * batch_size + min(i, remaining)) 
                for i in range(num_cores)]
        results = pool.starmap(run_batch_simulations, args)
    
    chunk_trajectories = [traj for batch in results for traj in batch]
    
    # Process trajectories
    print(f"Chunk {chunk_idx+1}: Processing {len(chunk_trajectories)} trajectories")
    uniform_time_points = time_points
    
    # Process in smaller sub-chunks to manage memory
    sub_chunk_size = 1000  # Process 1000 trajectories at a time
    for i in range(0, len(chunk_trajectories), sub_chunk_size):
        sub_chunk = chunk_trajectories[i:i+sub_chunk_size]
        
        # Prepare arguments for parallel processing
        args = [(traj_idx + start_idx + i, traj, p_a, p_b, p_c, uniform_time_points) 
                for traj_idx, traj in enumerate(sub_chunk)]
        
        # Process in parallel
        with mp.Pool(processes=num_cores) as pool:
            results = list(pool.imap(process_trajectory_continuous, args))
        
        # Save directly to files
        file_mode = 'a' if (chunk_idx > 0 or i > 0) else 'w'
        
        # Collect and save each type of data
        sl_data = [item for r in results for item in r[0]]
        sl_sq_data = [item for r in results for item in r[1]]
        sd_data = [item for r in results for item in r[2]]
        
        save_data_chunk(sl_data, f'stochastic_length_{num_trajectories}.dat', file_mode)
        save_data_chunk(sl_sq_data, f'stochastic_length_squared_{num_trajectories}.dat', file_mode)
        save_data_chunk(sd_data, f'stochastic_deviation_{num_trajectories}.dat', file_mode)
        
        # Clear memory
        del results, sl_data, sl_sq_data, sd_data
        gc.collect()
        
        print(f"  Processed and saved sub-chunk {i//sub_chunk_size + 1}/{(len(chunk_trajectories)+sub_chunk_size-1)//sub_chunk_size}")
    
    # Clear chunk memory
    del chunk_trajectories
    gc.collect()
    
    print(f"Chunk {chunk_idx+1} completed")

def main():
    print(f"Starting simulation with {num_trajectories} trajectories and {num_time_points} time points")
    
    # Calculate or load state probabilities
    P_A, P_B, P_C = calculate_probabilities_from_file()
    
    # Process in chunks to manage memory
    chunk_size = 10000  # Process 10,000 trajectories at a time
    num_chunks = (num_trajectories + chunk_size - 1) // chunk_size
    
    # Clear output files if they exist
    for filename in [f'stochastic_length_{num_trajectories}.dat', 
                     f'stochastic_length_squared_{num_trajectories}.dat', 
                     f'stochastic_deviation_{num_trajectories}.dat']:
        if os.path.exists(filename):
            os.remove(filename)
    
    # Process each chunk
    for chunk_idx in range(num_chunks):
        print(f"\nProcessing chunk {chunk_idx+1}/{num_chunks}")
        simulate_and_process_chunk(chunk_idx, chunk_size, P_A, P_B, P_C)
        
        # Force garbage collection
        gc.collect()
    
    print(f"\nAll {num_trajectories} trajectories processed and saved to files")

if __name__ == "__main__":
    main()
