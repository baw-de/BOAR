Quickstart
==========

This guide helps you get up and running quickly with the project.

Prerequisites and Installation
------------------------------

Make sure you have followed the installation instructions in the
:doc:`Installation </installation>` section of the documentation. This includes
setting up a Python environment, installing dependencies, and verifying the
installation.

Testing your setup
------------------

To check that BOAR and BASEMENT 4.3 work together on your machine before preparing your own case, run the
light BASEMENT benchmark with an analytical solution (a few minutes; see
:doc:`Shallow-Water Benchmarks </benchmarks_swe>`):

.. code-block:: bash

    uv run python benchmarks/basement_macdonald.py

It generates its own mesh, configuration and ground truth under ``output/benchmarks/swashes_macdonald/``,
calibrates the Manning coefficients of its friction regions and prints ``PASS`` when the final water depth
matches the analytical one.


Configuring BOAR
-----------------

Once the project directory is set up and dependencies are installed, you must
set up your configuration files and prepare your data for calibration.

Once the configuration files, paths and the loss function are set up, you must
setup the initial simulation configuration, discharge files and ground truth
data. The typical running instance has the following directory structure:

.. code-block:: text

   project_directory/
       ├── src/*
       ├── user_configs/
       │   ├── constants.py
       │   ├── function_loss.py
       │   └── user_options.yaml
       ├── support_files/
       │   ├── discharge_files/
       │   │   ├── Q001.txt
       │   │   └── Q002.txt
       │   └── discharge_dictionary.csv    # Maps discharge files to simulation configurations
       ├── ground_truth/
       │   ├── CASE_Q001_*.csv              # Ground truth data for CASE at discharge Q001
       │   ├── CASE_Q001_1.csv              # Ground truth data for CASE at discharge Q001 for point collection 1
       │   ├── CASE_Q001_2.csv              # Ground truth data for CASE at discharge Q001 for point collection 2
       │   └── CASE_Q002_*.csv              # Multiple files per case allowed. However, if two discharges are present in the ground truth, make sure that they are specified in the discharge dictionary and user_defined_configs/user_options.yaml files.
       ├── user_defined_configs/ # Details below
       │   ├── constants.py
       │   ├── function_loss.py
       │   └── user_options.yaml
       ├── meshes/
       │   ├── mesh1_.2dm
       │   └── mesh2_.2dm
       ├── simulation_configs/              # Original BASEMENT configuration, never modified by BOAR
       │   ├── model.json                   # SETUP.simulation_name names the output folder
       │   ├── simulation.json
       │   └── results.json
       ├── output/
       │   └── <simulation_name>/           # Run folder: copied JSON files, setup.h5, results.h5,
       │                                    # optimization_memory.csv, tried_vectors/
       └── boar.py

The layout above is the *single mode*: one case per run. To queue several scenarios and discharges, see
`Batch calibration`_. BOAR only reads the files in ``simulation_configs/``. Each run copies them
to ``output/<simulation_name>/``, and all simulations run on that copy. The name is BASEMENT's own
``simulation_name`` entry in the ``SETUP`` section of ``model.json``:

.. code-block:: json

   {
       "SETUP": {
           "simulation_name": "my_case",
           "...": "..."
       }
   }

In the copy, relative ``mesh_file`` and
``discharge_file`` paths are made absolute: they are resolved against ``simulation_configs/`` and then the
project root. If ``general_options.overwrite`` is ``True``, ``output/<simulation_name>/`` is deleted before the
run. If it is ``False``, the earlier ``optimization_memory.csv`` is kept and the calibration resumes. The
JSON files are always copied fresh from the originals, so the run starts from the original friction values.

Every run also writes ``output/calibration_summary.csv``, with one row per case: ``scenario``, ``discharge``,
one ``friction_<region>`` column per calibrated region (the best vector), ``best_error``, ``n_iter_best`` (the
evaluation at which the best error was found, counted in ``optimization_memory.csv``), ``device`` and
``status`` (``ok`` or ``failed``). A new run of the same case replaces its row.

Batch calibration
^^^^^^^^^^^^^^^^^
BOAR switches to *batch mode* when ``ground_truth/`` holds subfolders. Each subfolder, named
``SCENARIO_DISCHARGE`` (for example ``S1_Q001``; the scenario may contain underscores), is one calibration.
The files inside keep the usual naming, ``S1_Q001_*.csv``.

.. code-block:: text

   project_directory/
       ├── ground_truth/
       │   ├── S1_Q001/S1_Q001_1.csv
       │   ├── S1_Q002/S1_Q002_1.csv
       │   └── S2_Q001/S2_Q001_1.csv
       ├── simulation_configs/
       │   ├── model.json, simulation.json, results.json    # shared template (S1_Q001, S1_Q002)
       │   └── S2_Q001/                                      # own configuration of S2_Q001
       │       ├── model.json
       │       ├── simulation.json
       │       └── results.json
       └── output/
           ├── calibration_summary.csv
           ├── S1_Q001/
           │   ├── boar.log
           │   └── <simulation_name>/                        # run folder, as in single mode
           ├── S1_Q002/...
           └── S2_Q001/...

For each case BOAR uses ``simulation_configs/SCENARIO_DISCHARGE/`` when that folder exists. Otherwise it
replicates the flat template ``simulation_configs/model.json`` and replaces the discharge of its inflow
boundaries (``uniform_in``, ``froude_in``) with the case's discharge: the file
``<discharge_file_directory>/<DISCHARGE>.txt`` when ``simulation_options.discharge_file`` is ``True``, otherwise
the value of the code in ``support_files/discharge_dictionary.csv``. The originals are never changed. Before the
first case starts, BOAR prints which cases replicate the template, and with which discharge, followed by a
progress bar:

.. code-block:: text

   BOAR batch: 3 case(s), serial | workers: 1 CPU (8 physical cores each) | CPU pinning: off (single socket)
     NOTE: 2 case(s) have no folder in simulation_configs/ and replicate the template simulation_configs/model.json, with the inflow discharge replaced per case.
       S1_Q001  template   -> discharge Q001.txt
       S1_Q002  template   -> discharge Q002.txt
       S2_Q001  own config (simulation_configs/S2_Q001/) -> discharge Q001.txt
     Summary: output/calibration_summary.csv
   Cases:  33%|################                                | 1/3 [04:12<08:24]

A case that cannot run (missing discharge file, no configuration, ...) or that fails during the calibration is
logged, gets a ``failed`` row in the summary, and the queue continues with the next case.

Parallel cases, GPUs and CPU sockets
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
By default the cases run one after the other. ``batch_options`` in ``user_options.yaml`` runs several at the
same time, on CPU and GPU workers in any combination:

.. code-block:: yaml

   batch_options:
     'n_cpu_workers': 2        # cases at the same time on the CPU (backend of basement_options: omp/seq)
     'cpu_cores': -1           # physical cores of EACH CPU run; -1 = shared evenly
     'n_gpu_workers': 1        # cases at the same time on GPUs
     'gpu_backend': cudaC      # cuda or cudaC
     'preferred_device': gpu   # takes the next case when a GPU and a CPU worker are both free
     'pin_to_socket': True     # several CPU sockets: one socket per CPU run, GPU runs on their card's socket
     # 'gpu_numa_nodes': [0, 1]  # socket of each GPU, only if auto-detection fails

- **Cores** are physical cores; hyperthreads are not counted. With ``cpu_cores: -1`` the physical cores of the
  machine (of each socket when runs are pinned, less one core per GPU worker) are shared evenly between the CPU
  workers, and each run uses BASEMENT with that many threads. A larger explicit value is accepted with a
  warning.
- **GPU workers** run BASEMENT with ``gpu_backend`` and are spread over the cards that ``nvidia-smi`` lists
  (``CUDA_VISIBLE_DEVICES``); without ``nvidia-smi`` the card is left to BASEMENT.
- **CPU sockets**: the placement comes from capiQ (``src/cpu_topology.py``). On a machine with several sockets
  (NUMA nodes) each CPU run stays inside one socket on its own physical cores, so its threads and memory stay
  local, and each GPU run gets one core on the socket its card is attached to. Windows uses
  ``start /NODE /AFFINITY``; Linux uses ``numactl --physcpubind --localalloc`` (or ``taskset``). Multi-socket
  Linux has not been tested. With one socket nothing is pinned. To see what is detected:
  ``uv run python -c "from src import cpu_topology; cpu_topology.report()"``.
- Each case writes its own log to ``output/SCENARIO_DISCHARGE/boar.log``; the main log keeps the queue.
- Single mode (flat ``ground_truth/``) ignores ``batch_options`` and uses ``basement_options`` as before.

Interrupting and restarting
^^^^^^^^^^^^^^^^^^^^^^^^^^^
Ctrl+C stops the batch together with the BASEMENT runs it started. Every finished evaluation is already in the
case's ``optimization_memory.csv``; only the simulation that was running is lost. Run BOAR again with
``general_options.overwrite: False`` to continue:

- cases that had finished (their run folder holds ``calibration_finished.json``) are not run again; their row
  is kept in the summary;
- interrupted cases continue from their memory: evaluated vectors are not simulated again, and the ``boar``
  engine continues its budget (``max_tested_vectors`` and ``max_no_improvement`` count the iterations of the
  earlier run). The ``optuna`` engine also counts the earlier trials.

Both apply to single mode as well. With ``overwrite: True`` every case starts over.

Benchmarks on the subcritical MacDonald channel (one case replicates the template, one has its own
configuration):

- ``uv run python benchmarks/basement_macdonald.py --batch`` runs the cases serially, then with 2 CPU workers
  (add ``--gpu`` for 1 GPU + 1 CPU worker); every layout must give accurate depths and the same results.
- ``uv run python benchmarks/basement_macdonald.py --batch-restart`` interrupts a run like Ctrl+C (serially and
  with 2 CPU workers), checks that no BASEMENT program is left running, restarts it with ``overwrite: False``
  and checks that finished cases are skipped and the others end as the uninterrupted run.

The main configuration files are located in the ``user_configs`` directory.

Main configuration files:

- :doc:`User-defined configs <source/user_defined_configs>`: Define constants, loss function and optimization options

Functionalities
---------------

Discharge
^^^^^^^^^
BOAR can calibrate either a single hydraulic roughness coefficient for an entire discharge range
or individual coefficients for specific discharge values. This behaviour is controlled by the
`simulation_options.discharge_file_list` option in the `user_options.yaml` file.

If `simulation_options.discharge_file_list` contains multiple discharge files, the code calibrates
a single coefficient for the entire discharge range. If it contains only one discharge file, the code
calibrates a single coefficient for that specific discharge value.

The number of discharge files specified in `simulation_options.discharge_file_list` must match the
number of discharge files specified in the ground-truth file names.

Friction regions
^^^^^^^^^^^^^^^^
BOAR supports the calibration of single of multiple hydraulic roughness coefficients for specific regions
of the domain. This allows for a more detailed calibration, as different regions of the domain may have
different roughness characteristics. If the domain has multiple regions, it is also possible to calibrate
a single region and keep the other regions fixed.

This behaviour is controlled by the `optimization_variable_options.regions` option in the `user_options.yaml`
file. If `optimization_variable_options.regions` contains multiple regions, the code calibrates individual
coefficients for each region. If it contains only one region, the code calibrates a
single coefficient for that region.

Running the Calibration
-----------------------

Once all the configuration files are set up, you can run BOAR. It is recommended
to run BOAR from the command line, as this allows you to easily monitor the
log file and manage multiple instances. To run BOAR, navigate to the project
directory in your terminal and execute the following command:

.. code-block:: bash

   run_boar.bat      # Windows
   ./run_boar.sh     # Linux / macOS

Both scripts run ``boar.py`` with the Python of ``.venv`` (from ``uv sync`` or pip; otherwise the one on the PATH; set
``PYTHON_EXE`` to use another one) and pass their options on. BOAR itself generates ``setup.h5`` from
``model.json`` and runs the BASEMENT version set in ``basement_options``. Instead of the scripts, you can also run

.. code-block:: bash

   uv run boar.py      # or: python boar.py, with .venv activated

Calibration progress and detailed execution logs are recorded in ``boar.log`` by default. Alternatively, a custom log file can be specified using the following command: ``python boar.py --log-file [filename]``.
