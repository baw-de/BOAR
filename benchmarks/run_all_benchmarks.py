"""Run all benchmark modules in this folder."""

from __future__ import annotations

import importlib
import pkgutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path


def run_all() -> None:
    folder = Path(__file__).resolve().parent
    current_module = Path(__file__).stem
    main_functions = []

    for module_info in sorted(pkgutil.iter_modules([str(folder)]), key=lambda m: m.name):
        if module_info.name == current_module:
            continue

        module = importlib.import_module(module_info.name)
        main = getattr(module, "main", None)
        if callable(main):
            main_functions.append(main)

    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(fn) for fn in main_functions]
        for future in futures:
            future.result()


if __name__ == "__main__":
    run_all()
