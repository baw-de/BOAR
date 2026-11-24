Installation
============

BOAR runs from its project folder in a virtual environment ``.venv``. Its dependencies are listed in
``pyproject.toml``. The environment can be set up with `uv <https://docs.astral.sh/uv/>`_ (default), which
installs the pinned versions of ``uv.lock``, or with pip.

Prerequisites
--------------
BOAR needs Python 3.11 to 3.14 (h5py publishes no wheels for 3.15 yet). For the default setup, also install uv
(see the `uv installation guide <https://docs.astral.sh/uv/getting-started/installation/>`_); it downloads
a suitable Python itself if none is installed.

Installation Steps
------------------

1. **Download BOAR**: You can download the latest version of BOAR from the
   GitHub repository and create a local instance.

   .. code-block:: bash

      git clone <final_repository_url>

2. **Install BOAR dependencies**: Navigate to the BOAR directory and create ``.venv`` with
   one of the two options. The launchers ``run_boar.bat`` and ``run_boar.sh`` use this ``.venv``.

   *With uv (default)*, using the locked versions:

   .. code-block:: bash

      cd boar
      uv sync --no-dev      # leave out --no-dev to also install the development tools

   *With pip*, using the same dependencies from ``pyproject.toml`` (versions not pinned by
   ``uv.lock``; ``--group`` needs pip 25.1 or newer):

   .. code-block:: bash

      cd boar
      python -m venv .venv
      source .venv/bin/activate    # On Windows use: .venv\Scripts\activate
      pip install -e .             # runtime dependencies
      pip install --group dev      # development tools (optional)

3. **Point BOAR to BASEMENT**: BOAR runs BASEMENT's ``BMv4_setup``, ``BMv4_simulation`` and
   ``BMv4_results`` from the install folder of ``basement_version`` in ``basement_options``
   (default ``"4.3"``): ``C:\Program Files\BASEMENT <version>.0\bin`` on Windows and ``/usr/bin`` on Linux,
   where the BASEMENT Ubuntu package installs them:

   .. code-block:: bash

      sudo apt install ./<downloaded BASEMENT package>.deb   # Linux (Ubuntu)

   If BASEMENT is installed elsewhere, set ``BASEMENT_PATH`` in ``user_defined_configs/constants.py``
   to that folder (it overrides the folder of ``basement_version``).

4. **Verify Installation**: After the installation is complete, you can verify
   that BOAR and its dependencies load by running:

   .. code-block:: bash

      uv run boar.py --help        # uv setup
      python boar.py --help        # pip setup, with .venv activated
