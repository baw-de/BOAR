Installation
============

The installation of BOAR is straightforward and can be done using pip.
Follow the steps below to install BOAR in your Python environment.

Prerequisites
--------------
Make sure you have Python 3.13 or higher installed on your system. You can check your Python version by running:

.. code-block:: bash

   python --version

A virtual environment is recommended to manage dependencies. You can create and activate a virtual environment using the following commands:

.. code-block:: bash

   python -m venv boar_env
   source boar_env/bin/activate  # On Windows use: boar_env\Scripts\activate

Installation Steps
------------------

1. **Download BOAR**: You can download the latest version of BOAR from the
   GitHub repository and create a local instance.

   .. code-block:: bash

      git clone <final_repository_url>

2. **Install BOAR dependencies**: Navigate to the BOAR directory and install
   the required dependencies using pip:

   .. code-block:: bash

      cd boar
      pip install -r requirements.txt

3. **Point BOAR to BASEMENT**: BOAR runs BASEMENT's ``BMv4_setup``, ``BMv4_simulation`` and
   ``BMv4_results`` from the install folder of ``basement_version`` in ``basement_options``
   (default ``"4.3"``): ``C:\Program Files\BASEMENT <version>.0\bin`` on Windows and ``/usr/bin`` on Linux,
   where the BASEMENT Ubuntu package installs them:

   .. code-block:: bash

      sudo apt install ./<downloaded BASEMENT package>.deb   # Linux (Ubuntu)

   If BASEMENT is installed elsewhere, set ``BASEMENT_PATH`` in ``user_defined_configs/constants.py``
   to that folder (it overrides the folder of ``basement_version``).

4. **Verify Installation**: After the installation is complete, you can verify
   that BOAR is installed correctly by running:

   .. code-block:: bash

      python -c "import boar; print(boar.__version__)"
