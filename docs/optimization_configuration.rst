User Options Configuration
==========================

This guide explains the ``user_options.yaml`` configuration file used to control BOAR's behavior.

.. contents::
   :local:

General Options
---------------

The ``general_options`` section controls basic BOAR behavior.

.. code-block:: yaml

   general_options:
     'clear_start': True       # Re-creates output directory, deleting previous data
     'silent': True            # Do not print log messages
     'log_dev': False          # Print all log messages (including simulation logs)

BASEMENT Options
----------------

The ``basement_options`` section controls BASEMENT simulation settings.

.. code-block:: yaml

   basement_options:
     'cleanup': False          # Remove previous simulation results before running
     'backend': omp            # Backend options: seq, omp, cuda, cudaC, cudaO
     'nthreads': -1            # Number of cores (-1 = all available cores)

Backend Options
~~~~~~~~~~~~~~~

+----------+--------------------------------------------------+
| Option   | Description                                      |
+==========+==================================================+
| seq      | Sequential (single-threaded) execution           |
+----------+--------------------------------------------------+
| omp      | OpenMP (multi-threaded) execution                |
+----------+--------------------------------------------------+
| cuda     | CUDA execution                                   |
+----------+--------------------------------------------------+
| cudaC    | CUDA execution with sequential CPU processor     |
+----------+--------------------------------------------------+
| cudaO    | CUDA execution with OpenMP CPU processor         |
+----------+--------------------------------------------------+

Simulation Options
------------------

Configure discharge files and simulation inputs.

.. code-block:: yaml

   simulation_options:
     'discharge_file': True                                # Discharge defined by txt file
     'discharge_file_directory': /path/to/discharge_files  # Directory containing discharge files
     'discharge_file_list': ['Q001.txt', 'Q002.txt']        # List of discharge files to use

Optimization Variable Options
-----------------------------

The ``optimization_variable_options`` section defines the calibration parameters.

.. code-block:: yaml

   optimization_variable_options:
     'initial_vector': file   # Initial friction distribution (file, float, or list)
     'regions': ['floodplain']  # Regions to optimize (None = all regions)
     'constraints': None      # Parameter constraints (see below)
     'bounds': [!!python/tuple [10, 60]]  # Optimization bounds
     'precision': 1e-1        # Decimal precision (1, 0.1, 0.01, etc.)
     'save_errors': True      # Save error comparison points to CSV
     'save_tried_vectors': False  # Save results files for all tried vectors
     'in_house_optimization': False  # Use in-house optimizer (False = use Optuna)

Initial Vector Options
~~~~~~~~~~~~~~~~~~~~~~

+----------+-----------------------------------------------------------+
| Option   | Description                                               |
+==========+===========================================================+
| file     | Read initial friction from simulation file                |
+----------+-----------------------------------------------------------+
| float    | Use a constant value for initial friction                 |
+----------+-----------------------------------------------------------+
| list     | Use array of constant values (region_id, value)           |
+----------+-----------------------------------------------------------+

Constraints
~~~~~~~~~~~

It is recommended to avoid constraints as they may lead to local minima.

.. code-block:: yaml

   'constraints': {
       'expression': "x > y or np.isclose(x, y)",  # Constraint expression
       'variables': ['x', 'y']                       # Region/variable names
   }

Example expressions:

.. code-block:: python

   # Floodplain roughness > main channel roughness
   "x > y"

   # Floodplain roughness equals main channel roughness
   "np.isclose(x, y)"

   # Floodplain > main channel OR floodplain equals channel
   "x > y or np.isclose(x, y)"

Bounds
~~~~~~

Define optimization bounds for each region.

.. code-block:: yaml

   # Single region bounds
   'bounds': [!!python/tuple [10, 60]]

   # Multiple region bounds
   'bounds': [!!python/tuple [1, 12], !!python/tuple [2, 100]]

Sampling Options
----------------

The ``sampling_options`` section controls initial sampling strategies.

.. code-block:: yaml

   sampling_options:
     'max_lhs_runs': 200   # Number of samples for Latin Hypercube Sampling
     'seed': 42            # Random number generator seed (None for random)

Surrogate Model Options
-----------------------

The ``surrogate_model_options`` section controls the Gaussian Process Regression model.

.. code-block:: yaml

   surrogate_model_options:
     'opt_mem_override': False         # Override first memory check
     'n_initial': 10                  # Minimum samples before iterative optimization
     'max_tested_vectors': 50         # Maximum number of vectors to try
     'tolerance': 1e-4                # Stop criterion for optimization

   # In-house optimization options (when in_house_optimization: True)
     'test_population': 67956         # Number of samples for surrogate model
     'GPR_iterations': 500            # Maximum GPR iterations
     'max_no_improvement': 10         # Max iterations without improvement

Memory Override Options
~~~~~~~~~~~~~~~~~~~~~~~

+---------------------+----------------------------------------------------------+
| Option              | Description                                              |
+=====================+==========================================================+
| opt_mem_override    | Allow new samples even if memory exceeds n_samples       |
+---------------------+----------------------------------------------------------+

Full Configuration File
-----------------------

Below is the complete default configuration file:

.. literalinclude:: ../user_defined_configs/user_options.yaml
   :language: yaml
   :linenos:
   :caption: user_options.yaml
   :name: user_options_full