"""
This module is an example of a user-defined loss function for the Basement-Calibrator.
It computes the Root Mean Square Error (RMSE) between simulated and observed water depths at specified calibration points.
The function can be customized to include additional metrics, such as velocity errors, or to apply different weighting schemes.

It is important that the function accepts the following arguments:
    - `simulation_data`: A dictionary containing the simulation results, including cell centroids and hydrodynamic properties (e.g., water depth, velocity).
    - `ground_truth_data`: A dictionary containing the observed data at calibration points.

The function should return a single float value representing the loss, which the optimization algorithm will minimize.
"""
# Third-party imports
import numpy as np
import pandas as pd
from src import utils

def loss_function(
    simulation_data: dict,
    ground_truth_data: dict,
) -> float:
    """
    Extracts simulation data at calibration points and computes the RMSE for water depth.

    Args:
        simulation_data (dict): Contains simulation data (e.g., centroids and hydrodynamic properties).
        ground_truth_data (dict): Contains calibration points with known water depths.

    Returns:
        float: The sum of RMSE values for all calibration cross-sections.
        (pd.Series): Optional, a series of errors for each calibration point, which can be used for further analysis or visualization.
    """
    # Initialize an array to store RMSE values
    rmse_depth = []
    rmse_ux = []
    rmse_uy = []
    errors = pd.DataFrame(columns=['x', 'y', 'error_h', 'error_ux', 'error_uy'])

    for ground_truth in ground_truth_data:
        # Create an empty array for simulated cross-section data
        gt_numpy = ground_truth.to_numpy()
        sim_cs = {'cx':[], 'cy':[], 'h':[]}

        for i, (val_x, val_y, val_h) in enumerate(gt_numpy):
            # Find the closest simulation cells to the calibration point
            closest_cell_ids, cell_distance = utils.find_closest_cell(
                simulation_data['centroids'],
                (val_x, val_y),
                npoints=6,
                distance=True
            )

            # Interpolate water depth using IDW (Inverse Distance Weighting)
            sim_h = simulation_data['hyd']['h'][closest_cell_ids]
            sim_ux = simulation_data['hyd']['ux'][closest_cell_ids]
            sim_uy = simulation_data['hyd']['uy'][closest_cell_ids]

            sim_cs['h'].append(np.average(sim_h, weights=1 / (cell_distance**2)))
            sim_cs['ux'].append(np.average(sim_ux, weights=1 / (cell_distance**2)))
            sim_cs['uy'].append(np.average(sim_uy, weights=1 / (cell_distance**2)))

            sim_cs['cx'].append(val_x)
            sim_cs['cy'].append(val_y)

        # Checks for negative water depth
        if any(h < 0 for h in sim_cs['h']):
            return 1e6

        # Compute RMSE for this cross-section
        error_h = ground_truth['H [m]'] - sim_cs['h']
        error_ux = ground_truth['Ux [m/s]'] - sim_cs['ux']
        error_uy = ground_truth['Uy [m/s]'] - sim_cs['uy']

        # Store pointwise errors for potential further analysis or visualization
        new_rows = pd.DataFrame({
            'x': sim_cs['cx'],
            'y': sim_cs['cy'],
            'error_h': error_h,
            'error_ux': error_ux,
            'error_uy': error_uy
        })
        errors = pd.concat([errors, new_rows], ignore_index=True)

        # Compute RMSE values
        rmse_depth_i = np.sqrt(np.mean((ground_truth['H [m]'] - sim_cs['h'])**2))
        rmse_ux_i = np.sqrt(np.mean((ground_truth['Ux [m/s]'] - sim_cs['ux'])**2))
        rmse_uy_i = np.sqrt(np.mean((ground_truth['Uy [m/s]'] - sim_cs['uy'])**2))

        # Store RMSE values
        rmse_depth.append(rmse_depth_i)
        rmse_ux.append(rmse_ux_i)
        rmse_uy.append(rmse_uy_i)

    # Combine the RMSE values for each file
    mean_rmse_depth = np.mean(rmse_depth)
    mean_rmse_ux = np.mean(rmse_ux)
    mean_rmse_uy = np.mean(rmse_uy)

    loss_value = 0.5 * mean_rmse_depth + 0.25 * mean_rmse_ux + 0.25 * mean_rmse_uy

    ## It is possible to create a more complex loss function that combines multiple metrics (e.g., water depth and velocity errors) with different weights. The example above uses a simple weighted sum of RMSE values for water depth and velocity components.

    ## But it is also possible to return a simple loss value based on a single metric, such as water depth RMSE, if that is the primary focus of the calibration. In that case, the function would return only the RMSE for water depth:
    # loss_value = mean_rmse_depth

    ## The second return value is optional and can be used to save the error comparison points for all tried vectors, if specified in the user options.
    ## If the second value is not needed, simply return:
    # return loss_value, None

    return loss_value, errors