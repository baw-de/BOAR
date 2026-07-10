Quickstart
==========

This guide helps you get up and running quickly with the project.

Prerequisites and Installation
------------------------------

Make sure you have followed the installation instructions in the
:doc:`Installation </installation>` section of the documentation. This includes
setting up a Python environment, installing dependencies, and verifying the
installation.


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
       ├── simulation/
       └── boar.py

Several instances of BOAR can be run in parallel. It is recommended to use
different coding instances, such that the aforementioned directory structure
is duplicated for each instance. Each instance should have its own configuration
files, ground truth data, and simulation inputs. Some configuration files such
as the mesh directory and support files can be shared across instances.

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

   run_boar.bat

or

.. code-block:: bash

   python boar.py

Calibration progress and detailed execution logs are recorded in ``boar.log`` by default. Alternatively, a custom log file can be specified using the following command: ``python boar.py --log-file [filename]``.
