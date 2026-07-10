#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Utility functions for I/O operations and data manipulation.

This module provides functions for:
    - Logging messages
    - Modifying NumPy structured arrays
    - Exporting data to CSV/XLSX formats
    - Finding closest cells in a mesh
    - Creating constraint functions

Functions
---------
write_log : Log messages at different levels.
modify_structured_array : Rename/delete columns in arrays.
export_structured_array : Export arrays to CSV/XLSX.
find_closest_cell : Find closest mesh cell to a point.
simulation_cleanup : Remove simulation output files.
create_constraint_function : Create constraint functions.
"""

if __name__ == "__main__":
    raise Exception("This file must be run as a module")

# Standard library imports
import ast
import csv
import os
from collections import defaultdict

import numpy as np
import openpyxl


def write_log(logger, message: str, level: str = "info", silent: bool = False):
    """
    Logs the message using the specified level.

    Args:
        logger: The logger object to use for logging.
        message (str): The message to log.
        level (str): The logging level ('info', 'debug', 'error', etc.).
        silent (bool): If True, suppresses logging (except errors).
    """
    if level == "error":
        silent = False  # Always log errors

    if not silent:
        # This uses the level passed (like 'info', 'debug') to call the appropriate logging method.
        getattr(logger, level)(message)


def modify_structured_array(arr, rename_dict=None, delete_cols=None):
    """
    Modify a structured NumPy array by renaming and/or deleting columns.

    Args:
        arr (np.ndarray): The structured array to modify.
        rename_dict (dict, optional): Dictionary mapping old to new column names.
        delete_cols (list, optional): List of column names to delete.

    Returns:
        np.ndarray: The modified structured array.
    """

    # Check if input is a structured array
    if not arr.dtype.names:
        raise ValueError("Input must be a structured numpy array")

    # Start with the original array's dtype
    new_dtype = arr.dtype.descr.copy()

    # Create a mapping of old names to new names
    name_mapping = {}

    # Rename columns if rename_dict is provided
    if rename_dict:
        for old_name, new_name in rename_dict.items():
            if old_name in arr.dtype.names:
                # Update the name in the new dtype
                for i, (name, type_) in enumerate(new_dtype):
                    if name == old_name:
                        new_dtype[i] = (new_name, type_)
                        name_mapping[new_name] = old_name  # Create mapping

    # Delete columns if delete_cols is provided
    if delete_cols:
        for col in delete_cols:
            if col in arr.dtype.names:
                # Remove the column from the new dtype
                new_dtype = [dt for dt in new_dtype if dt[0] != col]

    # Create new structured array with updated dtype
    new_arr = np.empty(arr.shape, dtype=new_dtype)

    # Copy data from the old array to the new array using the mapping
    for new_name in new_arr.dtype.names:
        old_name = name_mapping.get(new_name, new_name)  # Get old name from mapping
        if old_name in arr.dtype.names:  # Ensure the old name exists
            new_arr[new_name] = arr[old_name]

    return new_arr


def print_and_export_column_names(arr):
    """
    Print and return column names from a structured array.

    Args:
        arr (np.ndarray): A structured NumPy array.

    Returns:
        set: Set of column names.
    """
    column_names = arr.dtype.names
    # Print each column name
    for name in column_names:
        print(name)

    # Convert the column names to a list
    column_names_set = set(column_names)

    return column_names_set


def export_structured_array(array, file_path):
    """
    Export a structured NumPy array to a CSV or XLSX file.

    Args:
        array (np.ndarray): The structured array to export.
        file_path (str): The full output file path.

    Raises:
        ValueError: If input is not a structured array or format is unsupported.
    """
    if not isinstance(array, np.ndarray):
        raise ValueError("The input array must be a NumPy ndarray.")

    if array.dtype.names is None:
        raise ValueError("The input array must be a structured NumPy array with named fields.")

    # Determine the file extension
    _, file_extension = os.path.splitext(file_path)

    # Create the directory if it does not exist
    directory = os.path.dirname(file_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)

    if file_extension.lower() == ".csv":
        with open(file_path, mode="w", newline="") as csv_file:
            writer = csv.writer(csv_file)
            # Write the header
            writer.writerow(array.dtype.names)
            # Write the data
            for row in array:
                writer.writerow(row)

    elif file_extension.lower() == ".xlsx":
        workbook = openpyxl.Workbook()
        sheet = workbook.active

        # Write the header
        sheet.append(array.dtype.names)
        # Write the data
        for row in array:
            sheet.append(row.tolist())

        workbook.save(file_path)

    else:
        raise ValueError("Unsupported file format. Use '.csv' or '.xlsx'.")


def find_closest_cell(
    centroids: np.ndarray, point: tuple, npoints: int = 3, distance: bool = False
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """
    Find the closest cell to a given point based on centroid coordinates.

    Args:
        centroids (np.ndarray): 2D array of shape (n, 2) with x, y coordinates.
        point (tuple): Coordinates (x, y) of the reference point.
        npoints (int): Number of closest cells to find.
        distance (bool): If True, returns distances along with indices.

    Returns:
        int or tuple: Index of closest cell, or (indices, distances) if distance=True.

    Example:
        >>> centroids = np.array([[0, 0], [1, 1], [2, 2]])
        >>> find_closest_cell(centroids, (1.5, 1.5))
        array([2])
    """
    # Convert the input point to a NumPy array
    point_array = np.array(point)

    # Extract the x and y coordinates of the centroids
    cx = centroids["cx"]
    cy = centroids["cy"]

    cxy = np.column_stack((cx, cy))

    # Compute the squared Euclidean distances between the point and all centroids
    distances = np.sum((cxy - point_array) ** 2, axis=1)

    # Find the index of the minimum distance
    closest_indices = np.argsort(distances)[:npoints]

    if distance:
        return closest_indices, distances[closest_indices]

    return closest_indices


def simulation_cleanup(simulation_folder: str) -> None:
    """
    Clean up the simulation folder by deleting result files.

    Args:
        simulation_folder (str): Path to the simulation folder.

    Returns:
        None
    """
    delete_files = ["results.h5", "results_aux.h5"]

    for filename in delete_files:
        file_path = os.path.join(simulation_folder, filename)
        if os.path.exists(file_path):
            os.remove(file_path)


def create_constraint_function(expr: str, variables: list):
    """
    Create a constraint function from a user-provided logical expression.

    Args:
        expr (str): The logical expression (e.g., "x > y").
        variables (list): Names of variables in the expression.

    Returns:
        function: A function that evaluates the constraint.

    Example:
        >>> f = create_constraint_function("x > y", ['x', 'y'])
        >>> f([2, 1])
        True
    """
    # Parse the expression into an AST
    parsed_expr = ast.parse(expr, mode="eval")

    # Define allowed node types for validation
    allowed_nodes = {
        ast.Expression,
        ast.BoolOp,
        ast.Compare,
        ast.Name,
        ast.Load,
        ast.Call,
        ast.And,
        ast.Or,
        ast.Not,
        ast.Is,
        ast.IsNot,
        ast.Eq,
        ast.NotEq,
        ast.Lt,
        ast.LtE,
        ast.Gt,
        ast.GtE,
        ast.Attribute,
        ast.Constant,
        # Arithmetic operations (BinOp)
        ast.BinOp,
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.Pow,
        ast.Mod,
        ast.FloorDiv,
        ast.UAdd,
        ast.USub,  # Unary operations
    }

    # Python < 3.14 compatibility: include deprecated ast.NameConstant when present.
    if hasattr(ast, "NameConstant"):
        allowed_nodes.add(ast.NameConstant)

    # Walk through the AST and validate nodes
    for node in ast.walk(parsed_expr):
        if type(node) not in allowed_nodes:
            raise ValueError(f"Disallowed node type: {type(node).__name__}")
        # ast.Call is in the allowlist for numpy functions (e.g. np.clip, np.abs).
        # Validate that any call targets only `np.<attr>` — no builtins or other names.
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Attribute):
                # Allow only np.<something>
                if not (isinstance(func.value, ast.Name) and func.value.id == "np"):
                    raise ValueError(
                        f"Disallowed function call: only numpy (np.*) functions are permitted, "
                        f"got '{ast.unparse(func)}'"
                    )
            elif isinstance(func, ast.Name):
                raise ValueError(
                    f"Disallowed function call: '{func.id}' is not permitted. "
                    "Only numpy functions (np.*) may be called."
                )

    # Define the constraint function
    def constraint_function(values):
        if len(values) != len(variables):
            raise ValueError(f"Expected {len(variables)} values, got {len(values)}")

        # Map variables to their corresponding values
        local_vars = dict(zip(variables, values))

        # Evaluate the compiled expression
        # nosec B307 – safe: parsed_expr was validated against an AST allowlist above;
        # only arithmetic, comparison and boolean nodes are permitted, and function calls
        # are restricted to np.* (numpy) attributes only.
        return eval(compile(parsed_expr, filename="<string>", mode="eval"), {"np": np}, local_vars)  # nosec B307

    return constraint_function


def nested_defaultdict():
    """
    Create a nested defaultdict.

    Returns:
        defaultdict: A defaultdict with dict as the default factory.
    """
    return defaultdict(dict)
