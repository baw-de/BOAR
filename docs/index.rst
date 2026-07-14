BOAR documentation
==================

We present *Bayesian Optimization for Automated Roughness calibration in two-dimensional hydrodynamic models* (BOAR),
an open-source Python framework for automated hydraulic roughness
calibration in two-dimensional shallow-water models using BASEMENT. The tool
combines Bayesian optimization with conditional sampling to incorporate expert
knowledge, such as feasible parameter ranges and inter-parameter constraints,
reducing the number of costly model evaluations. BOAR standardizes calibration
workflows, improves reproducibility, and supports flexible user-defined loss
functions.


.. toctree::
   :maxdepth: 2
   :caption: Contents

   quickstart
   installation
   optimization_configuration
   loss_function
   benchmarks
   source/modules


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
