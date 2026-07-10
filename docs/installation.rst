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

3. **Verify Installation**: After the installation is complete, you can verify
   that BOAR is installed correctly by running:

   .. code-block:: bash

      python -c "import boar; print(boar.__version__)"
