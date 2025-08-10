import pandas as pd
import numpy as np
from scipy.stats import weibull_min
import matplotlib.pyplot as plt

# ----------------------------
# 1. Load Processed Data
# ----------------------------
# ---- Load your structured CSV ----
df = pd.read_csv('extrapolated_data.csv')  

# ---- Prepare storage for results ----
all_results = []

# ---- Get unique iterations ----
iterations = df['iteration'].unique()

for iter_num in iterations:
    iter_df = df[df['iteration'] == iter_num].copy()
    times = iter_df['time_min'].values

    

    # ---- Fit Weibull to time data for this iteration ----
    # Use all points as "life data" for demonstration
    # shape, loc, scale = weibull_min.fit(times, floc=0)
    # beta, eta = shape, scale
    # Update parameters after each new data point
    for index, row in iter_df.iterrows():
        current_data = iter_df.loc[:index]
        beta, eta = weibull_min.fit(current_data['time_min'], floc=0)[0], weibull_min.fit(current_data['time_min'], floc=0)[2]
        iter_df.loc[index, 'RUL'] = eta * (-np.log(0.2))**(1/beta) - row['time_min']

    pdf = weibull_min.pdf(times, beta, scale=eta)
    cdf = weibull_min.cdf(times, beta, scale=eta)

    plt.figure(figsize=(10, 5))
    # plt.plot(times, pdf, label='PDF (actual times)', color='deepskyblue')
    plt.plot(times, cdf, label='CDF (actual times)', color='orange')
    plt.xlabel('Time to Failure (Observed)')
    plt.ylabel('Probability')
    plt.title(f'Weibull TTF Distribution (Iteration {iter_num+1})')
    plt.legend()
    plt.grid(True)
    # plt.show()



    # ---- Estimate RUL at each point ----
    # Here, RUL is time until cumulative failure probability reaches 20% (change as needed)
    iter_df['RUL'] = [eta * (-np.log(0.2))**(1/beta) - t for t in times]

    # ---- Maintenance flag: HI >= 0.8 or RUL < 60 min ----
    iter_df['maintenance'] = (iter_df['health_indicator'] >= 0.8) | (iter_df['RUL'] < 60)

    # ---- Store results ----
    all_results.append(iter_df)

    # ---- Visualization: Dual Y-Axis for HI and RUL ----
    fig, ax1 = plt.subplots(figsize=(10, 5))

    color = 'tab:blue'
    ax1.set_xlabel('Time (min)')
    ax1.set_ylabel('Health Indicator (HI)', color=color)
    ax1.plot(iter_df['time_min'], iter_df['health_indicator'], color=color, label='HI')
    ax1.axhline(0.8, color='red', linestyle='--', label='HI Threshold')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_ylim(0, 1.1)

    # Plot RUL on right y-axis
    ax2 = ax1.twinx()
    color = 'tab:green'
    ax2.set_ylabel('Remaining Useful Life (min)', color=color)
    ax2.plot(iter_df['time_min'], iter_df['RUL'], color=color, label='RUL')
    ax2.axhline(60, color='orange', linestyle='--', label='RUL Threshold (1 hr)')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_ylim(0, iter_df['RUL'].max() * 1.1)

    # Maintenance triggers
    ax1.scatter(
        iter_df.loc[iter_df['maintenance'], 'time_min'],
        iter_df.loc[iter_df['maintenance'], 'health_indicator'],
        color='magenta', label='Maintenance Trigger', zorder=5
    )

    # Legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=8)

    plt.title(f'Iteration {iter_num+1}: HI and RUL Trends (Dual Axis)')
    plt.tight_layout()
    

    plt.show()

    # ---- Print Weibull parameters for this iteration ----
    print(f"Iteration {iter_num+1}: Weibull β={beta:.2f}, η={eta:.2f}")

# ---- Combine all results and save ----
results_df = pd.concat(all_results, ignore_index=True)
results_df.to_csv('spm_results_per_iteration.csv', index=False)
print("Results saved to 'spm_results_per_iteration.csv'")



# ----------------------------
# 2. Extract PM/CM Times per Iteration
# ----------------------------
def get_maintenance_times(results_df):
    maintenance_data = []
    iterations = results_df['iteration'].unique()
    
    for iter_num in iterations:
        iter_df = results_df[results_df['iteration'] == iter_num]
        
        # Get first PM time
        pm_events = iter_df[iter_df['maintenance']]
        pm_time = pm_events['time_min'].min() if not pm_events.empty else np.nan
        # print(pm_time)
        # Get CM time (last timestamp)
        cm_time = iter_df['time_min'].max()
        # print(cm_time)
        # print()
        maintenance_data.append({
            'iteration': iter_num+1,
            'pm_time': pm_time,
            'cm_time': cm_time
        })
    
    return pd.DataFrame(maintenance_data)

# ----------------------------
# 3. Fixed Period Cost Calculation
# ----------------------------
def calculate_fixed_period_costs(maintenance_df, fixed_period_hours=100):
    C_p = 1000  # Preventive maintenance cost
    C_c = 1250  # Corrective maintenance cost
    C_d = 300   # Downtime cost per hour
    
    fixed_period_min = fixed_period_hours * 60
    results = []
    
    for _, row in maintenance_df.iterrows():
        # SPM Cycle (PM-based)
        if not np.isnan(row['pm_time']):
            pm_cycle = row['pm_time'] + 60  # 30min downtime
            num_pm = fixed_period_min // pm_cycle
            remaining = fixed_period_min % pm_cycle
            spm_cost = (num_pm * C_p) + (num_pm * 0.5 * C_d)
        else:
            num_pm = 0
            spm_cost = 0
        
        # Traditional Cycle (CM-based)
        cm_cycle = row['cm_time'] + 120  # 2hr downtime
        num_cm = fixed_period_min // cm_cycle
        trad_cost = (num_cm * C_c) + (num_cm * 2 * C_d)
        
        results.append({
            'iteration': row['iteration'],
            'spm_cycles': num_pm,
            'trad_cycles': num_cm,
            'spm_cost': spm_cost,
            'trad_cost': trad_cost
        })
    
    return pd.DataFrame(results)

# ----------------------------
# 4. Visualization
# ----------------------------
def plot_cost_comparison(cost_df):
    fig, ax = plt.subplots(1, 2, figsize=(18, 6))
    
    # Per iteration comparison
    cost_df.plot.bar(x='iteration', y=['spm_cost', 'trad_cost'], 
                    ax=ax[0], color=['#1f77b4', '#ff7f0e'])
    ax[0].set_title('Maintenance Costs per Iteration')
    ax[0].set_ylabel('Cost ($)')
    ax[0].grid(axis='y', alpha=0.3)
    
    # Total cost comparison
    totals = cost_df[['spm_cost', 'trad_cost']].sum()
    totals.plot.bar(ax=ax[1], color=['#1f77b4', '#ff7f0e'])
    ax[1].set_title(f'Total Costs Over {fixed_period_hours} Hours')
    ax[1].set_ylabel('Total Cost ($)')
    ax[1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# ----------------------------
# 5. Execute Analysis
# ----------------------------
# Set fixed period (default: 100 hours)
fixed_period_hours = 100

# Process data
maintenance_times = get_maintenance_times(results_df)
cost_comparison = calculate_fixed_period_costs(maintenance_times, fixed_period_hours)

# Show results
print("Maintenance Cycle Analysis:")
print(maintenance_times)
print("\nCost Comparison:")
print(cost_comparison)
print(f"\nTotal SPM Cost: ${cost_comparison['spm_cost'].sum():.2f}")
print(f"Total Traditional Cost: ${cost_comparison['trad_cost'].sum():.2f}")
print(f"Savings: ${cost_comparison['trad_cost'].sum() - cost_comparison['spm_cost'].sum():.2f}")

# Visualize
plot_cost_comparison(cost_comparison)

