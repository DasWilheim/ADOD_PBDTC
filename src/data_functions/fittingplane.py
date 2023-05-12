import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import interp2d
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

DDD =  True  # 3D plot or contour plot
SAVE = True  # stores the fittted plane in a pickle file
# Read the data from a csv file
data = pd.read_csv("C:\\Users\\wille\\OneDrive\\Documenten\\Werk\\Sparqle\\Coding\\AMoD\\src\\data_functions\\data_result\\data_with_travel_time.csv")

x = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
y = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0]

v1 = data["Total_time_between_order_and_delivery_prio1"]
v2 = v1.to_numpy()
v3 = np.reshape(v2, (len(y), len(x)))

# Prepare the input data for linear regression
X, Y = np.meshgrid(x, y)
XY = np.vstack((X.flatten(), Y.flatten())).T
Z = v3.flatten()

poly = PolynomialFeatures(degree=4)
XY_poly = poly.fit_transform(XY)

# Fit the linear regression model
model = LinearRegression()
model.fit(XY_poly, Z)

# Predict the Z values using the fitted model
Z_pred = model.predict(XY_poly).reshape(X.shape)

# Specify the x-value and desired height
x_value = 0.75  # Example x-value
desired_height = 27

# Create a set of possible y-values
y_values = np.linspace(min(y), max(y), 1000)

# Compute the polynomial features for the given x-value and possible y-values
x_grid, y_grid = np.meshgrid(np.array([x_value]), y_values)
xy_grid = np.vstack((x_grid.flatten(), y_grid.flatten())).T
xy_poly = poly.transform(xy_grid)

# Predict the Z values using the fitted model
z_pred = model.predict(xy_poly)

# Find the y-value for which the predicted Z is closest to the desired height
index = np.argmin(np.abs(z_pred - desired_height))
y_value = y_values[index]

print(f"The y-value corresponding to x={x_value} and height={desired_height} is {y_value:.2f}")

if SAVE:
    with open('C:\\Users\\wille\\OneDrive\\Documenten\\Werk\\Sparqle\\Coding\\AMoD\\src\\data_functions\\data_plane\\fitted_plane.pickle', 'wb') as f:
        pickle.dump({'model': model, 'poly': poly, 'x': x, 'y': y, 'X': X, 'Y': Y, 'z_pred': z_pred}, f)

if DDD: 
    # Create a 3D plot of the original data
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X, Y, v3, color='b', label='Original Data')

    # Plot the fitted plane
    ax.plot_surface(X, Y, Z_pred, color='r', alpha=0.3, label='Fitted Plane')

    # Set axis labels and title
    ax.set_xlabel('Packages of type 1 (in percentage %)')
    ax.set_ylabel('Priority ratio')
    ax.set_zlabel('Elapsed time between order and delivery type 1')
    ax.set_title('Fitted Plane for Elapsed Time Data')

    plt.show()

else:
    # Create the contour plot of the fitted plane
    fig, ax = plt.subplots()
    contour = ax.contour(X, Y, Z_pred, cmap='viridis')
    plt.colorbar(contour)


    # Add the point (x_value, y_value) with the desired height to the contour plot
    # ax.scatter(x_value, y_value, marker='o', color='red', label=f'Height {desired_height}')

    # Set the axis labels and title
    ax.set_xlabel('Packages of type 1 (in percentage %)')
    ax.set_ylabel('Priority ratio')
    ax.set_title('Height Lines on Fitted Plane')

    plt.show()