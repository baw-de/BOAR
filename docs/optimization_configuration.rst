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
     'clear_start': True       # [bool] Re-create output directory on start
     'silent': True            # [bool] Suppress log messages
     'log_dev': False          # [bool] Print all logs including simulation logs

BASEMENT Options
----------------

The ``basement_options`` section controls BASEMENT simulation settings.

.. code-block:: yaml

   basement_options:
     'cleanup': False          # [bool] Remove previous results before running
     'backend': omp            # [str] Backend: seq, omp, cuda, cudaC, cudaO
     'nthreads': -1            # [int] Number of cores (-1 = all available)

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
     'discharge_file': True                                # [bool] Define the model setup discharge using a TXT file. (Default: False)
     'discharge_file_directory': /path/to/discharge_files  # [str] Directory containing discharge files. Only required if 'discharge_file' is True.
     'discharge_file_list': ['Q001.txt', 'Q002.txt']       # [list] List of discharge files to use. Only required if 'discharge_file' is True.

Optimization Variable Options
-----------------------------

The ``optimization_variable_options`` section defines the calibration parameters.

.. code-block:: yaml

   optimization_variable_options:
     'initial_vector': file                 # [str] file, float, or list. (Default: file)
     'regions': ['floodplain']              # [list] Regions to optimize. (Default: all friction regions)
     'constraints': None                    # [dict] Parameter constraints. (Default: None, see below)
     'bounds': None                         # [list] Optimization bounds. (Default: None)
     'precision': 1e-1                      # [float] Decimal precision. (Default: 1)
     'save_errors': True                    # [bool] Save error points to CSV. (Default: False)
     'save_tried_vectors': False            # [bool] Save all results files. (Default: False)
     'opt_engine': False                    # [str] Optimization engine ("boar" or "optuna"). Default: "boar"

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
     'tolerance': 1e-4                 # [float] Stop criterion
     'n_initial': 5                    # [int] Min samples before optimization. (Default: 5)
     'max_tested_vectors': 100         # [int] Max vectors to try. (Default: 100)
     'opt_mem_override': False         # [bool] Override memory check. (Default: False)

   # Only used if 'opt_engine' is set to "boar"
     'test_population': 67956            # [int] Surrogate model samples
     'max_no_improvement': 10            # [int] Max iterations without progress (Default: 100)
     'GPR_iterations': 500               # [int] The number of restarts of the optimizer for finding the kernel’s parameters which maximize the log-marginal likelihood. (Default: 500)
     'GPR_alpha': 1e-6                   # [float] Value added to the diagonal of the kernel matrix during fitting. This can prevent a potential numerical issue during fitting, by ensuring that the calculated values form a positive definite matrix. (Default: 1e-6)
     'EI_exploration-exploitation': 0.01 # [float] Exploration-exploitation parameter for Expected Improvement acquisition function. (Default: 0.01)

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
