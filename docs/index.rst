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


How to Cite
==================

If you use BOAR in your research, please cite:

    de Oliveira, L.E.D., Franca, M.J., Huber, N.P., Vanzo, D., 2026. BOAR: Bayesian
    optimization for automated roughness calibration in two-dimensional hydrodynamic
    models. SoftwareX 35, 102899. https://doi.org/10.1016/j.softx.2026.102899

BibTeX:

.. code-block:: bibtex

   @article{DEOLIVEIRA2026102899,
   title = {BOAR: Bayesian optimization for automated roughness calibration in two-dimensional hydrodynamic models},
   journal = {SoftwareX},
   volume = {35},
   pages = {102899},
   year = {2026},
   issn = {2352-7110},
   doi = {https://doi.org/10.1016/j.softx.2026.102899},
   url = {https://www.sciencedirect.com/science/article/pii/S2352711026003900},
   author = {Luiz E.D. {de Oliveira} and Mário J. Franca and Nils P. Huber and Davide Vanzo},
   keywords = {Bayesian optimization, Hydraulic roughness calibration, Shallow water solvers, BASEMENT, Surrogate model, Gaussian process, Model calibration},
   abstract = {BOAR is an open-source Python framework for automated hydraulic roughness calibration in two-dimensional shallow-water numerical solvers. BOAR combines Bayesian optimization with conditional sampling to integrate expert knowledge, including feasible parameter ranges and inter-parameter constraints, thereby reducing the number of model evaluations. It standardizes calibration workflows, improves reproducibility, and supports flexible, user-defined loss functions. Applied to 25 laboratory vegetated compound-channel flow cases, it converged within a median of 9 model runs for a depth-error of 0.4%--1.5% tolerance. The current implementation interfaces with BASEMENT and can be extended to other river-modeling tools, providing a flexible and extensible solution for hydraulic modeling.}
   }


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
