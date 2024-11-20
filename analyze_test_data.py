import math
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

def get_csv_files(folder):
    """
    Retrieves all CSV files in the specified folder that match the pattern 'hate_score_results{index}.csv'.
    """
    files = [os.path.join(folder, f) for f in os.listdir(folder)
             if f.startswith('hate_score_results') and f.endswith('.csv')]
    return files

def read_data(files):
    """
    Reads and concatenates data from a list of CSV files into a single DataFrame.
    """
    df_list = [pd.read_csv(f) for f in files]
    data = pd.concat(df_list, ignore_index=True)
    return data

def compute_counts(data, x_values, y_values):
    """
    Calculates the number of lines remaining for each combination of x and y thresholds.
    """
    counts = np.zeros((len(x_values), len(y_values)), dtype=int)
    text_lengths = data['text'].astype(str).apply(len)
    hate_scores = data['hate_score'].astype(float)

    for i, x in enumerate(x_values):
        for j, y in enumerate(y_values):
            # Apply filter: len(text) >= x and hate_score >= y
            mask = (text_lengths >= x) & (hate_scores >= y)
            counts[i, j] = mask.sum()
    return counts

def compute_new_metric(data, x_values, y_values):
    """
    Computes the new metric for each combination of x and y thresholds.
    Metric = log(Average Hate Score) * Average Text Length * log(Number of Lines Remaining)
    """
    metric = np.full((len(x_values), len(y_values)), np.nan)
    text_lengths = data['text'].astype(str).apply(len)
    hate_scores = data['hate_score'].astype(float)

    for i, x in enumerate(x_values):
        for j, y in enumerate(y_values):
            # Apply filter: len(text) >= x and hate_score >= y
            mask = (text_lengths >= x) & (hate_scores >= y)
            num_lines = mask.sum()

            if num_lines > 500 and num_lines < 3000:
                avg_hate_score = hate_scores[mask].mean()
                avg_text_length = text_lengths[mask].mean()
                # Ensure avg_hate_score and num_lines are greater than zero before taking log
                if avg_hate_score > 0:
                    metric[i, j] = math.log(avg_hate_score)*math.log(avg_text_length) * num_lines
                else:
                    metric[i, j] = np.nan
            else:
                metric[i, j] = np.nan  # No data points for this threshold
    return metric

def plot_counts(counts, x_values, y_values):
    """
    Generates a contour plot of the counts with specified color mapping.
    """
    X, Y = np.meshgrid(x_values, y_values, indexing='ij')
    Z = counts
    max_count = counts.max()
    min_count = counts.min()

    # Define levels and colors
    levels = [min_count, 500, 3000, max_count]
    colors_list = ['red', 'green', 'yellow']

    # Adjust levels and colors if max_count is less than 5000
    if max_count <= 1500:
        levels = [min_count, max_count]
        colors_list = ['red']
    elif max_count <= 5000:
        levels = [min_count, 1500, max_count]
        colors_list = ['red', 'green']

    # Create custom colormap
    cmap = colors.ListedColormap(colors_list)
    norm = colors.BoundaryNorm(levels, cmap.N)

    plt.figure(figsize=(10, 8))
    contour = plt.contourf(X, Y, Z, levels=levels, cmap=cmap, norm=norm)
    plt.colorbar(contour, label='Number of Lines Remaining', ticks=levels)
    plt.xlabel('Minimum Text Length (x)')
    plt.ylabel('Minimum Hate Score (y)')
    plt.title('Lines Remaining After Applying Filters')
    plt.tight_layout()
    plt.show()

def plot_new_metric(metric, x_values, y_values):
    """
    Generates a contour plot of the new metric and highlights the global maximum.
    """
    X, Y = np.meshgrid(x_values, y_values, indexing='ij')
    Z = metric

    plt.figure(figsize=(6, 5))
    # Use a colormap that highlights differences
    contour = plt.contourf(X, Y, Z, levels=50, cmap='plasma')
    plt.colorbar(contour, label='Metric Value')
    plt.xlabel('Minimum Text Length (x)')
    plt.ylabel('Minimum Hate Score (y)')
    plt.title('Metric: log(Avg Hate Score) * log(Avg Text Length) * Num Lines')

    # Find the indices of the maximum value
    max_idx = np.unravel_index(np.nanargmax(Z), Z.shape)
    max_x = x_values[max_idx[0]]
    max_y = y_values[max_idx[1]]
    max_value = Z[max_idx]

    # Plot the red dot at the global maximum
    plt.plot(max_x, max_y, 'ro', markersize=10, label='Global Maximum')
    plt.legend()
    plt.tight_layout()
    plt.show()

    print(f"Global maximum at x={max_x}, y={max_y} with value={max_value:.2f}")

    return max_x, max_y, max_value

def filter_and_save_data(data, x_threshold, y_threshold):
    """
    Filters the data based on the thresholds and saves it to 'test/filtered_hate_score_results.csv'
    """
    # Apply filter: len(text) >= x_threshold and hate_score >= y_threshold
    text_lengths = data['text'].astype(str).apply(len)
    hate_scores = data['hate_score'].astype(float)
    mask = (text_lengths >= x_threshold) & (hate_scores >= y_threshold)
    filtered_data = data[mask]

    # Ensure the 'test' folder exists
    output_folder = 'Test_Data'
    os.makedirs(output_folder, exist_ok=True)
    output_file = os.path.join(output_folder, 'filtered_hate_score_results.csv')

    # Save the filtered data
    filtered_data.to_csv(output_file, index=False)
    print(f"Filtered data saved to {output_file}")

def main():
    folder = 'Test_Data'  # Adjust this path if your data is in a different folder
    files = get_csv_files(folder)
    data = read_data(files)
    x_values = list(range(0, 201))  # x from 0 to 200 inclusive
    y_values = list(range(0, 101, 1))  # y from 0 to 100 inclusive in steps of 1

    counts = compute_counts(data, x_values, y_values)
    metric = compute_new_metric(data, x_values, y_values)

    # Plot the original counts
    plot_counts(counts, x_values, y_values)

    # Plot the new metric and get the global maximum thresholds
    max_x, max_y, max_value = plot_new_metric(metric, x_values, y_values)

    # Filter data based on global maximum thresholds and save
    filter_and_save_data(data, max_x, max_y)

if __name__ == "__main__":
    main()
