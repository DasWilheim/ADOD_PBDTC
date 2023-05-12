############ HEATMAPS ############
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp2d


# Read the data from a csv file
data = pd.read_csv("C:\\Users\\wille\\OneDrive\\Documenten\\Werk\\Sparqle\\Coding\\AMoD\\src\\data_functions\\data_result\\data_with_travel_time.csv")
# Generate some sample data with irregular spacing
# y = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0]
x = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
# x = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9]
y = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0]

# Create the heatmap data
v1 = data["complete_orders_percentage"]
v2 = v1.to_numpy()
v3 = np.reshape(v2, (len(y),len(x)))
print(v3.shape)

# Interpolate the data
resolution = 12
f = interp2d(x, y, v3, kind='cubic')
x_new = np.linspace(min(x), max(x), len(x)*resolution -(resolution-1))
y_new = np.linspace(min(y), max(y), len(y)*resolution -(resolution-1))
v3_interpolated = f(x_new, y_new)

# Create the heatmap using imshow
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))

# Set the tick labels to match the irregularly spaced data
selected_ticks_x = range(0, len(x), 3)
selected_ticks_y = range( 0, len(y), 5)
selected_labels_x = [x[i] for i in selected_ticks_x]
selected_labels_y = [y[i] for i in selected_ticks_y]


im1 = axes[0].imshow(v3, origin='lower', cmap='hot')
axes[0].set_xticks(selected_ticks_x)
axes[0].set_yticks(selected_ticks_y)
axes[0].set_xticklabels(selected_labels_x)
axes[0].set_yticklabels(selected_labels_y)
cbar1 = fig.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
axes[0].set_xlabel('Packages of type 1 (in percentage %)')
axes[0].set_ylabel('Priority ratio')
axes[0].set_title('Original: Percentage of complete orders')



selected_ticks_x2 = range(0, len(x_new), 3 * resolution)
selected_ticks_y2 = range( 0, len(y_new), 5 * resolution)
selected_labels_x2 = [x_new[i] for i in selected_ticks_x2]
selected_labels_y2 = [y_new[i] for i in selected_ticks_y2]
# Plot the interpolated heatmap
im2 = axes[1].imshow(v3_interpolated, origin='lower', cmap='hot')
axes[1].set_xticks(selected_ticks_x2)
axes[1].set_yticks(selected_ticks_y2)
axes[1].set_xticklabels(selected_labels_x2)
axes[1].set_yticklabels(selected_labels_y2)
cbar2 = fig.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
axes[1].set_xlabel('Packages of type 1 (in percentage %)')
axes[1].set_ylabel('Priority ratio')
axes[1].set_title('Interpolated: Percentage of complete orders')

plt.tight_layout()
plt.show()