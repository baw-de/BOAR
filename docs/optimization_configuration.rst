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
     'overwrite': True         # [bool] Delete output/<simulation_name>/ before the run (False: resume from its optimization_memory.csv)
     'silent': True            # [bool] Suppress log messages
     'log_dev': False          # [bool] Print all logs including simulation logs

BASEMENT Options
----------------

The ``basement_options`` section controls BASEMENT simulation settings.

.. code-block:: yaml

   basement_options:
     'backend': omp            # [str] Backend: seq, omp, cuda, cudaC, cudaO
     'nthreads': -1            # [int] Number of cores (-1 = all available)
     'basement_version': "4.3" # [str] BASEMENT version: "4.2" or "4.3" (Default: "4.3")

BASEMENT Version
~~~~~~~~~~~~~~~~

``basement_version`` tells BOAR how to read the bed elevation from ``results.h5``,
which BASEMENT stores differently from version to version:

+---------+---------------------------------------------------------------+
| Version | Bed elevation in ``results.h5``                               |
+=========+===============================================================+
| 4.2     | ``CellsAll/BottomEl`` (one static dataset)                    |
+---------+---------------------------------------------------------------+
| 4.3     | ``RESULTS/CellsAll/BottomEl/<time>`` (one dataset per output) |
+---------+---------------------------------------------------------------+

The same setting selects the BASEMENT programs BOAR runs: on Windows those in
``C:\Program Files\BASEMENT <version>.0\bin`` (e.g. ``C:\Program Files\BASEMENT 4.3.0\bin``), on Linux those in
``/usr/bin``. Setting ``BASEMENT_PATH`` in ``user_defined_configs/constants.py`` overrides the folder;
it must then hold the version set here.
If it does not match the results, BOAR stops with an error that names the version
the file appears to come from.

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

Discharge
---------

The discharge needs no option. BOAR takes the discharge code from the ground-truth file names
(``CASE_Q001_1.csv`` -> ``Q001``) and the kind of discharge from the inflow boundary (``uniform_in`` or
``froude_in``) of ``model.json``:

- ``discharge_file``: the time series ``discharge_files/<CODE>.txt``
  (``DISCHARGE_FILES_FOLDER`` in ``constants.py``);
- ``discharge``: the constant value of ``<CODE>`` in ``discharge_files/discharge_dictionary.csv`` (needed only in this case).

Earlier versions set this with ``simulation_options`` (``discharge_file``, ``discharge_file_directory``,
``discharge_file_list``). Those options are no longer needed: ``discharge_file_directory`` still replaces the
folder above, with a warning, and the other two are ignored.

Optimization Variable Options
-----------------------------

The ``optimization_variable_options`` section defines the calibration parameters.

.. code-block:: yaml

   optimization_variable_options:
     'initial_vector': file                 # [str] file, float, or list. (Default: file)
     'regions': ['floodplain']              # [list] Regions to optimize. (Default: all friction regions)
     'constraints': None                    # [dict] Parameter constraints. (Default: None, see below)
     'bounds': None                         # [list] Optimization bounds. (Default: None)
     'precision': 1                         # [float] Decimal precision. (Default: 1)
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
     'test_population': None             # [int] Surrogate model candidate samples. None = auto-sized (see below). (Default: None)
     'max_test_population': 100000       # [int] Cap on the auto-sized test population. (Default: 100000)
     'max_no_improvement': 10            # [int] Max iterations without progress (Default: 100)
     'GPR_iterations': 500               # [int] The number of restarts of the optimizer for finding the kernel’s parameters which maximize the log-marginal likelihood. (Default: 500)
     'GPR_alpha': 1e-6                   # [float] Value added to the diagonal of the kernel matrix during fitting. This can prevent a potential numerical issue during fitting, by ensuring that the calculated values form a positive definite matrix. (Default: 1e-6)
     'EI_exploration-exploitation': 0.01 # [float] Exploration-exploitation parameter for Expected Improvement acquisition function. (Default: 0.01)

Test Population
~~~~~~~~~~~~~~~

The test population is the fixed set of candidate vectors on which the Expected Improvement is evaluated.
With ``'test_population': None`` (default) its size follows from the bounds, the precision and the constraints:

* every precision grid point inside the bounds that satisfies the constraints is a candidate. For example,
  two regions with bounds ``[10, 60]``, precision ``1`` and the constraint ``x >= y`` give
  ``51 * 52 / 2 = 1326`` candidates;
* if the feasible grid holds more than ``max_test_population`` points, a random (seeded) subset of
  ``max_test_population`` points is used;
* if the grid is too large to enumerate (more than 1,000,000 points before constraints, or more than
  ``max_test_population`` if that is larger), or no precision is given, ``max_test_population`` samples
  are drawn with Latin Hypercube Sampling.

An integer ``test_population`` draws that many Latin Hypercube samples instead.

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
