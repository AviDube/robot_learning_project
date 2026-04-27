import matplotlib.pyplot as plt
import seaborn as sns
from tbparse import SummaryReader
import pandas as pd

# --- CONFIGURATION: Modify these names and paths ---
runs = {
    "PPO New Rewards + Full Dim": "./ppo_franka_tb/PPO_44",
    "PPO New Rewards + Reduced Dim": "./ppo_franka_tb/PPO_46",
    "SAC New Rewards + Full Dim": "./ppo_franka_tb/SAC_5",
    "SAC New Rewards + Reduced Dim": "./ppo_franka_tb/SAC_6",
}

# Metric settings
TARGET_TAG = 'eval/mean_reward'
WINDOW_SIZE = 20  # Increase this for smoother lines, decrease for more detail
SAVE_FILENAME = "franka_performance_comparison.png"

def plot_tensorboard_comparison():
    # 1. Setup the visual style
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(12, 6)) # Wider figure to accommodate the legend on the right

    # 2. Iterate and plot
    for label, path in runs.items():
        try:
            reader = SummaryReader(path)
            df = reader.scalars
            
            # Filter for the specific metric
            run_data = df[df['tag'] == TARGET_TAG]
            
            if not run_data.empty:
                # Apply smoothing (Moving Average)
                smoothed_values = run_data['value'].rolling(window=WINDOW_SIZE, min_periods=1).mean()
                
                # Plotting the line
                plt.plot(run_data['step'], smoothed_values, label=label, linewidth=2.5)
            else:
                print(f"Warning: Tag '{TARGET_TAG}' not found in {path}")
                
        except Exception as e:
            print(f"Error loading {path}: {e}")

    # 3. Formatting the Axes and Labels
    plt.title(f"Comparison: {TARGET_TAG}", fontsize=15, fontweight='bold', pad=20)
    plt.xlabel("Training Steps", fontsize=12)
    plt.ylabel("Mean Reward", fontsize=12)

    # 4. Moving the Legend OUTSIDE the plot area
    # bbox_to_anchor=(1.02, 1) moves it slightly to the right of the axes
    plt.legend(
        title="Experiment Runs", 
        loc='lower right', 
        frameon=True,
        shadow=True,
        fontsize='small',
        title_fontsize='medium'
    )

    # Adjust X-axis to scientific notation (e.g., 1e7)
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))

    # 5. Save and Show
    # bbox_inches='tight' is critical; it ensures the legend doesn't get cut off when saving
    plt.tight_layout()
    plt.savefig(SAVE_FILENAME, dpi=300, bbox_inches='tight')
    print(f"Success! Plot saved as {SAVE_FILENAME}")
    plt.show()

if __name__ == "__main__":
    plot_tensorboard_comparison()