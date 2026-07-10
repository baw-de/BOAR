
Loss Function
=============

Overview
--------

The optimisation loss function is customisable and user-defined. The user must provide a loss function
that accepts two positional input dictionaries:

1. **Simulation data** — containing the mesh cell centroids and all hydrodynamic results.
2. **Ground-truth data** — containing the observed values at fixed XY coordinates.

This design allows the use of any tabular format for the ground-truth data. Any resolved hydrodynamic
variable (e.g., water depth, flow velocities, Reynolds stresses, etc.) can be used individually or
combined to compose the loss value. Utility tools such as ``utils/find_closest_cell`` assist the user
in matching observed points to simulated cells.

Return Values
-------------

The user-defined loss function must return:

- **Required:** A single floating-point value representing the global loss.
- **Optional:** A second output containing a list of point-wise errors.

The optional point-wise errors do not affect the calibration process and are stored in CSV files solely
for visualisation purposes.

Although not directly implemented, users may define variable-specific error thresholds within the loss
function that, once satisfied, return a null loss value and terminate the code execution.

Example:

.. code-block:: python

   if water_depth_error < 0.1 and velocity_error < 0.1:
       return 0.0, pointwise_errors

Or implement weighted loss components that prioritise certain variables:

.. code-block:: python

   loss = 0.7 * water_depth_error + 0.3 * velocity_error
   return loss, pointwise_errors


Example :doc:`Loss Function <source/user_defined_configs>` implementation:
----------------------------------------------------------------------------------------

.. literalinclude:: ../user_defined_configs/function_loss.py
   :language: python
   :pyobject: loss_function


Input Dictionary Structure
--------------------------

``simulation_data``
~~~~~~~~~~~~~~~~~~~

+---------------------------+-------------------------------------------------------------+--------------------------------------------------+
| Variable                  | Description                                                 | In ``simulation_data``, accessible with:         |
+===========================+=============================================================+==================================================+
| water_depth               | Array of simulated water depths per cell                    | ``simulation_data['hyd']['h']``                  |
+---------------------------+-------------------------------------------------------------+--------------------------------------------------+
| velocity_x                | Array of X-component flow velocities per cell               | ``simulation_data['hyd']['ux']``                 |
+---------------------------+-------------------------------------------------------------+--------------------------------------------------+
| velocity_y                | Array of Y-component flow velocities per cell               | ``simulation_data['hyd']['uy']``                 |
+---------------------------+-------------------------------------------------------------+--------------------------------------------------+
| specific_discharge_x      | Array of X-component specific discharge per cell            | ``simulation_data['hyd']['qx']``                 |
+---------------------------+-------------------------------------------------------------+--------------------------------------------------+
| specific_discharge_y      | Array of Y-component specific discharge per cell            | ``simulation_data['hyd']['qy']``                 |
+---------------------------+-------------------------------------------------------------+--------------------------------------------------+
| turbulent_k               | Array of turbulent kinetic energy per cell                  | ``simulation_data['turb_k']``                    |
+---------------------------+-------------------------------------------------------------+--------------------------------------------------+
| turbulent_reynolds_xx     | Array of turbulent Reynolds stresses in X direction         | ``simulation_data['turb_reynolds'][:, 0]``       |
+---------------------------+-------------------------------------------------------------+--------------------------------------------------+
| turbulent_reynolds_yy     | Array of turbulent Reynolds stresses in Z direction         | ``simulation_data['turb_reynolds'][:, 1]``       |
+---------------------------+-------------------------------------------------------------+--------------------------------------------------+


``ground_truth_data``
~~~~~~~~~~~~~~~~~~~~~

The ground-truth dictionary can hold any tabular structure with a header. Note that the coordinates must be retrievable as ``ground_truth_data['observations'][i]['x']`` and ``ground_truth_data['observations'][i]['y']`` for each observation point ``i``. The observed variable values can be accessed similarly, e.g., ``ground_truth_data['observations'][i]['water_depth']``.

Utility: ``utils/find_closest_cell``
-------------------------------------

The ``find_closest_cell`` utility maps an observed point (X, Y) to the index of the nearest
simulated mesh cell, based on the cell centroid coordinates stored in ``simulation_data``.

.. code-block:: python

   cell_index = find_closest_cell(simulation_data, x, y)

This allows the user to extract the simulated hydrodynamic variables at the location closest
to each observation point.

Notes
-----

- The loss function file must be placed at ``user_defined_configs/function_loss.py``.
- The function name must be ``loss_function``.
- Only the first return value (global loss) influences the calibration algorithm.
- The optional second return value (point-wise errors) is exported as a CSV file for post-processing
  and visualisation only.
- Users may implement early stopping by returning ``0.0`` as the loss value when predefined
  error thresholds are met for all variables of interest.
- Further information can be found at :doc:`Function Loss <source/user_defined_configs>`.
