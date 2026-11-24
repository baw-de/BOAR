# Third-party notices

BOAR is licensed under the GNU General Public License v3.0 (see [LICENSE](LICENSE)). It includes code from the
project below, under its own license.

## capiQ

- Project: capiQ: Computational Automation Pipeline for Hydro-morphodynamic simulation Queues, version 0.2.0
- Repository: https://github.com/KIT-IWU-HydroEng/capiQ
- DOI: https://doi.org/10.5281/zenodo.24505638
- Copyright (c) 2026 Karlsruhe Institute of Technology (KIT)
- License: MIT (text below)

Used in BOAR:

| BOAR file | From capiQ | Use |
|---|---|---|
| `src/cpu_topology.py` | `src/capiq/cpu_topology.py` | copied; only the imports of `report()` adapted |
| `tests/test_cpu_topology.py` | `tests/test_cpu_topology.py` | tests copied; those of capiQ's own queue and logs left out |
| `src/batch.py` (`choose_device`, `kill_process_trees`, the parallel queue loop) | `src/capiq/calculate_sim_queue.py`, `src/capiq/utils.py` | adapted |

```text
MIT License

Copyright (c) 2026 Karlsruhe Institute of Technology (KIT)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
