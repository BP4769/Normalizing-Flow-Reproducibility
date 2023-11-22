import numpy as np

def caret(line_length, angle, num_points, std_dev):
    # Calculate the slopes of the lines
    slope = np.tan(angle)

    # Calculate the number of points for each line
    num_points_per_line = num_points // 2

    # Generate points for the first line
    x1 = np.linspace(0, line_length / 2, num_points_per_line)
    y1 = -slope * x1

    # Generate points for the second line (mirror image)
    x2 = np.linspace(-line_length / 2, 0, num_points_per_line)
    y2 = slope * x2


    # Concatenate the x and y coordinates of the points from both lines
    x = np.concatenate((x1, x2))
    y = np.concatenate((y1, y2))

    # Add Gaussian noise to the x and y coordinates
    x_noisy = x + np.random.normal(0, std_dev, x.shape)
    y_noisy = y + np.random.normal(0, std_dev, y.shape)

    return x_noisy, y_noisy


def generate_circular_data(num_points, radius, noise_factor):
    # Generate angles uniformly
    angles = np.linspace(0, 2*np.pi, num_points)
    
    # Create points in polar coordinates
    x = radius * np.cos(angles)
    y = radius * np.sin(angles)
    
    # Add noise to the points
    x += np.random.normal(0, noise_factor, num_points)
    y += np.random.normal(0, noise_factor, num_points)
    
    return x, y