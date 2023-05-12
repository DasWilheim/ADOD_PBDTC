from matplotlib import animation
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

DDD = False   # 3D plot or contour plot

# Read the data from a csv file
data = pd.read_csv("data_result/data_with_travel_time.csv")

x = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
y = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0]

v1 = data["Total_time_between_order_and_delivery_prio1"]
v2 = v1.to_numpy()
v3 = np.reshape(v2, (len(y), len(x)))

# Prepare the input data for linear regression
X, Y = np.meshgrid(x, y)
XY = np.vstack((X.flatten(), Y.flatten())).T
Z = v3.flatten()

poly = PolynomialFeatures(degree=3)
XY_poly = poly.fit_transform(XY)

# Fit the linear regression model
model = LinearRegression()
model.fit(XY_poly, Z)

# Predict the Z values using the fitted model
Z_pred = model.predict(XY_poly).reshape(X.shape)
# Set your desired values here
frames = 100
x_value = np.linspace(min(x), 0.85, frames)
desired_height = 27.2

# Generate dummy data for the animation (replace this with your actual data)
Z_pred_animation = np.repeat(Z_pred[np.newaxis, :, :], frames, axis=0)
x_animation = x_value
y_animation = []


for i in range(frames):
    # Create a set of possible y-values
    y_values = np.linspace(min(y), max(y), 1000)

    # Compute the polynomial features for the given x-value and possible y-values
    x_grid, y_grid = np.meshgrid(np.array([x_value[i]]), y_values)
    xy_grid = np.vstack((x_grid.flatten(), y_grid.flatten())).T
    xy_poly = poly.transform(xy_grid)

    # Predict the Z values using the fitted model
    z_pred = model.predict(xy_poly)

    # Find the y-value for which the predicted Z is closest to the desired height
    index = np.argmin(np.abs(z_pred - desired_height))
    y_value = y_values[index]
    y_animation.append(y_value)

# animation
def anim(frames_Z):
    def init():
        contour = ax.contour(X, Y, frames_Z[0], cmap='viridis')
        height_point.set_data([x_animation[0]], [y_animation[0]])
        return contour, height_point

    def animate(n):
        for coll in ax.collections:
            coll.remove()
        contour = ax.contour(X, Y, frames_Z[n], cmap='viridis')
        height_point.set_data([x_animation[n]], [y_animation[n]])
        return contour, height_point

    fig, ax = plt.subplots(figsize=(10, 6))
    plt.colorbar(ax.contour(X, Y, frames_Z[0], cmap='viridis'))
    ax.set_xlabel('Packages of type 1 (in percentage %)')
    ax.set_ylabel('Priority ratio')
    ax.set_title('Height Lines on Fitted Plane with Desired Height Point')

    height_point, = ax.plot([], [], marker='o', color='red', label=f'Height {desired_height}')
    ax.legend()

    anime = animation.FuncAnimation(fig, animate, init_func=init, frames=len(frames_Z), interval=75)
    return anime

# Save the animation
print('Saving animation...')
anime = anim(Z_pred_animation)
anime.save('height_lines_animation.mp4', dpi=100, fps=None, extra_args=['-vcodec', 'libx264'])
print('Animation saved.')
