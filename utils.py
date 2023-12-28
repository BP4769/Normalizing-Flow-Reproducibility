import numpy as np
import matplotlib.pyplot as plt
import torch

def generate_continuously_colored_samples(num_samples):
    """
    Generate a set of points sampled from a bivariate standard normal distribution, with a continuously changing color 
    assigned to each point based on the y-value, using a colormap analogous to a rainbow.

    :param num_samples: Number of samples to generate.
    :return: A tuple (samples, colors), where samples is a 2D array of shape (num_samples, 2), each row representing
             a sample [x, y], and colors is a list of colors corresponding to each sample.
    """
    # Sample x and y coordinates from standard normal distributions
    x_samples = torch.randn(num_samples)
    y_samples = torch.randn(num_samples)
    samples = torch.stack((x_samples, y_samples), dim=1)
    # x_samples = np.random.normal(0, 1, num_samples)
    # y_samples = np.random.normal(0, 1, num_samples)
    # samples = np.column_stack((x_samples, y_samples))


    # Use a colormap to map y-values to colors
    colormap = plt.cm.rainbow
    color_norm = plt.Normalize(y_samples.min(), y_samples.max())
    colors = colormap(color_norm(y_samples))

    return samples, colors

def generate_grid_data(x_range, y_range, x_tick, y_tick):
    """
    Generate a grid of points in the x-y plane.

    :param x_range: A tuple (x_min, x_max) specifying the range of x values.
    :param y_range: A tuple (y_min, y_max) specifying the range of y values.
    :param x_tick: Number of points to generate along the x-axis.
    :param y_tick: Number of points to generate along the y-axis.
    :return: A 2D tensor of shape (x_tick * y_tick, 2), each row representing a point [x, y].
    """
    # Generate x and y values for the grid
    x_values = torch.linspace(x_range[0], x_range[1], x_tick)
    y_values = torch.linspace(y_range[0], y_range[1], y_tick)
    
    # Create grid points
    grid_points = torch.cartesian_prod(x_values, y_values)
    
    return grid_points