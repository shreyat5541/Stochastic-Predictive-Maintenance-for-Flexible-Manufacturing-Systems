

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# ========== CONFIGURATION ==========
file_path = 'mill_data2.csv'        # Path to your data file
voltage = 480                      # Voltage for load calculation
vib_failure = 1                  # Vibration failure threshold (mm/s)
# load_failure = 3.0    
load_failure = 1.0               # Load failure threshold (kW)
temp_failure = 70.0                # Temperature failure threshold (°C)
target_points = 100                # Total points per iteration (interp + extra)
extrapolation_ratio = 0.5          # 50% time extension beyond original data
poly_degree = 2                    # Polynomial degree for extrapolation
poly_points = 5                    # Number of last points to use for polynomial fitting

# ========== DATA LOADING & PROCESSING ==========
try:
    # df = pd.read_csv(file_path, nrows=109)
    df = pd.read_csv(file_path)
    print(f"Loaded {len(df)} rows from {file_path}")
except FileNotFoundError:
    raise FileNotFoundError(f"{file_path} not found")

# Rename and process columns
df = df.rename(columns={'smcDC': 'current', 'vib_spindle': 'vibration', 'time': 'time_raw'})
df['time_raw'] = df['time_raw'].astype(float)
df['iteration'] = (df['time_raw'].diff() < 0).cumsum()
df['time_min'] = df.groupby('iteration')['time_raw'].cumsum()

# Calculate derived metrics
df['load_kw'] = (df['current'] * voltage) / 1000
operating_hours = df['time_min'] / 60
df['temp_degc'] = 0.8 * operating_hours**2 + 2 * operating_hours + 25

base_df = df[['iteration', 'time_min', 'load_kw', 'vibration', 'temp_degc']]



def extrapolate_iteration_poly_preserve(df_iter, extra_points=15, extrapolation_ratio=0.5, poly_degree=2, poly_points=5):
    """
    Preserves original data and appends only new extrapolated points.
    Ensures iteration-specific time handling.
    """
    x = df_iter['time_min'].values
    y_vib = df_iter['vibration'].values
    y_load = df_iter['load_kw'].values
    y_temp = df_iter['temp_degc'].values

    # Fit polynomials on last few points (iteration-specific)
    points_to_use = min(poly_points, len(x))
    x_poly = x[-points_to_use:]
    vib_poly = np.poly1d(np.polyfit(x_poly, y_vib[-points_to_use:], poly_degree))
    load_poly = np.poly1d(np.polyfit(x_poly, y_load[-points_to_use:], poly_degree))
    temp_poly = np.poly1d(np.polyfit(x_poly, y_temp[-points_to_use:], poly_degree))

    # Calculate extended time range (iteration-specific)
    original_max = x.max()
    original_duration = original_max - x.min()
    extended_max = original_max + original_duration * extrapolation_ratio

    # Generate new timestamps starting exactly from the last original point
    if extra_points > 0:
        x_extra = np.linspace(original_max, extended_max, extra_points+1)[1:]  # Start from max, exclude first point
        vib_extra = vib_poly(x_extra)
        load_extra = load_poly(x_extra)
        temp_extra = temp_poly(x_extra)
        is_extrapolated = np.ones(len(x_extra), dtype=bool)

        extra_df = pd.DataFrame({
            'time_min': x_extra,
            'vibration': vib_extra,
            'load_kw': load_extra,
            'temp_degc': temp_extra,
            'iteration': int(df_iter['iteration'].iloc[0]),
            'is_extrapolated': is_extrapolated
        })
    else:
        extra_df = pd.DataFrame(columns=['time_min', 'vibration', 'load_kw', 'temp_degc', 'iteration', 'is_extrapolated'])

    # Mark original data
    orig_df = df_iter.copy()
    orig_df['is_extrapolated'] = False

    # Concatenate original and extrapolated data
    combined = pd.concat([orig_df, extra_df], ignore_index=True)
    return combined





def analyze_iterations_preserve(base_df, iterations=None, extra_points=15, 
                              extrapolation_ratio=0.5, poly_degree=2, poly_points=5):
    all_iterations = sorted(base_df['iteration'].unique())
    iterations = iterations or all_iterations
    
    extrapolated_dfs = []
    processed_iterations = []
    
    for i in iterations:
        iter_data = base_df[base_df['iteration'] == i]
        if len(iter_data) >= 2:
            extrapolated = extrapolate_iteration_poly_preserve(
                iter_data,
                extra_points=extra_points,
                extrapolation_ratio=extrapolation_ratio,
                poly_degree=poly_degree,
                poly_points=min(poly_points, len(iter_data))
            )
            extrapolated_dfs.append(extrapolated)
            processed_iterations.append(i)
    
    return pd.concat(extrapolated_dfs, ignore_index=True), processed_iterations




def add_cumulative_time(extrapolated_df, base_df):
    """Calculate cumulative time based ONLY on original data durations"""
    # Get original max times from raw data (not extrapolated)
    original_max_times = base_df.groupby('iteration')['time_min'].max()
    
    # Calculate offsets based on original durations
    cumulative_offsets = original_max_times.cumsum().shift(fill_value=0)
    
    # Apply offsets to extrapolated data
    extrapolated_df['time_cumulative'] = extrapolated_df.apply(
        lambda row: row['time_min'] + cumulative_offsets.get(row['iteration'], 0),
        axis=1
    )
    return extrapolated_df, cumulative_offsets


def add_health_indicator(df):
    """Calculate health indicator based on multiple metrics"""
    df['health_indicator'] = (
        0.45 * df['vibration'] / vib_failure +
        0.3 * df['load_kw'] / load_failure +
        0.25 * df['temp_degc'] / temp_failure
    )
    return df

# ========== VISUALIZATION FUNCTIONS ==========
def plot_iteration_breakdown(orig_df, extrapolated_df, iteration):
    """Detailed plot showing original vs extrapolated data with polynomial fit"""
    orig_data = orig_df[orig_df['iteration'] == iteration]
    extra_data = extrapolated_df[extrapolated_df['iteration'] == iteration]
    
    plt.figure(figsize=(12, 6))
    
    # Original data
    plt.plot(orig_data['time_min'], orig_data['vibration'], 'o', 
             color='blue', markersize=8, label='Original Data')
    
    # Split extrapolated data into interp/extra sections
    cutoff = orig_data['time_min'].max()
    mask_interp = extra_data['time_min'] <= cutoff
    mask_extra = ~mask_interp
    
    # Interpolated section (within original range)
    plt.plot(extra_data[mask_interp]['time_min'], 
             extra_data[mask_interp]['vibration'], 
             '-', color='green', lw=2, label='Interpolated')
    
    # Extrapolated section (polynomial fit)
    plt.plot(extra_data[mask_extra]['time_min'], 
             extra_data[mask_extra]['vibration'], 
             '--', color='red', lw=2, label='Polynomial Extrapolation')
    
    # Future points markers
    plt.scatter(extra_data[mask_extra]['time_min'], 
                extra_data[mask_extra]['vibration'], 
                color='orange', s=80, label='Extrapolated Points')
    
    plt.axvline(cutoff, color='black', ls='--', lw=1.5, label='Extrapolation Start')
    plt.title(f"Iteration {iteration}: Vibration Forecast with Polynomial Fit")
    plt.xlabel('Time (minutes)')
    plt.ylabel('Vibration (mm/s)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    # plt.show()

def plot_health_indicators(extrapolated_df, iterations=None):
    """Plot health indicator evolution with extrapolated data"""
    if iterations is None:
        iterations = sorted(extrapolated_df['iteration'].unique())
    
    plt.figure(figsize=(12, 6))
    for i in iterations:
        iter_data = extrapolated_df[extrapolated_df['iteration'] == i]
        
        # Split into original and extrapolated sections
        mask_orig = ~iter_data['is_extrapolated']
        mask_extra = iter_data['is_extrapolated']
        
        # Plot original section
        plt.plot(iter_data[mask_orig]['time_cumulative'], 
                 iter_data[mask_orig]['health_indicator'], 
                 '-', label=f'Iteration {i} (Original)')
        
        # Plot extrapolated section
        plt.plot(iter_data[mask_extra]['time_cumulative'], 
                 iter_data[mask_extra]['health_indicator'], 
                 '--', label=f'Iteration {i} (Extrapolated)')
    
    plt.axhline(y=0.8, color='r', linestyle='--', label='Maintenance Threshold (0.8)')
    plt.xlabel('Cumulative Time (min)')
    plt.ylabel('Health Indicator')
    plt.title('Health Indicator Evolution with Polynomial Extrapolation')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

def plot_all_iterations_vibration(orig_df, extrapolated_df, iterations=None):
    """Plot vibration across all iterations with extrapolated data"""
    if iterations is None:
        iterations = sorted(extrapolated_df['iteration'].unique())
    
    plt.figure(figsize=(12, 6))
    
    for i in iterations:
        # Original data with cumulative time
        orig = orig_df[orig_df['iteration'] == i]
        if 'time_cumulative' in orig.columns:
            plt.plot(orig['time_cumulative'], orig['vibration'], 'o', 
                     alpha=0.7, label=f'Original (Iter {i})')
        
        # Extrapolated data
        extra = extrapolated_df[extrapolated_df['iteration'] == i]
        
        # Split into original and extrapolated sections
        mask_orig = ~extra['is_extrapolated']
        mask_extra = extra['is_extrapolated']
        
        # Plot original range
        plt.plot(extra[mask_orig]['time_cumulative'], 
                 extra[mask_orig]['vibration'], 
                 '-', alpha=0.7)
        
        # Plot extrapolated range
        plt.plot(extra[mask_extra]['time_cumulative'], 
                 extra[mask_extra]['vibration'], 
                 '--', alpha=0.7)
    
    plt.xlabel('Cumulative Time (min)')
    plt.ylabel('Vibration (mm/s)')
    plt.title('Vibration Across All Iterations with Polynomial Extrapolation')
    plt.grid(True, alpha=0.3)
    
    # Create simplified legend
    handles, labels = plt.gca().get_legend_handles_labels()
    if len(handles) > 6:
        plt.legend(handles[:6], labels[:6], loc='upper left')
    else:
        plt.legend(loc='upper left')
    
    plt.show()

# ========== MAIN EXECUTION ==========
if __name__ == "__main__":
    # Process iterations (modify as needed)
    iterations_to_process = None  # Set to None for all iterations or specify a list like [0, 1, 2]
    
    processed_df, used_iters = analyze_iterations_preserve(
        base_df,
        iterations=iterations_to_process,
        extra_points=15,
        extrapolation_ratio=extrapolation_ratio,
        poly_degree=poly_degree,
        poly_points=poly_points
    )
    
    # Add cumulative time and health metrics
    processed_df, cumulative_offsets = add_cumulative_time(processed_df, base_df)
    processed_df = add_health_indicator(processed_df)



    
    # Add cumulative time to original df for comparison plots
    df['time_cumulative'] = df.apply(
        lambda row: row['time_min'] + cumulative_offsets.get(row['iteration'], 0),
        axis=1
    )
    
    # Save results
    processed_df.to_csv('polynomial_extrapolated_data.csv', index=False)
    print(f"Saved extrapolated data with polynomial fit to polynomial_extrapolated_data.csv")
    
    # Plot individual iterations with polynomial extrapolation
    for iter_num in used_iters:
        plot_iteration_breakdown(base_df, processed_df, iter_num)
    
    # Plot health indicators
    plot_health_indicators(processed_df, iterations=used_iters)
    
    # Plot all iterations together
    plot_all_iterations_vibration(df, processed_df, iterations=used_iters)
