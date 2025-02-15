"""
Functions to assist main.py in handling arguments.

References
----------
* https://stackoverflow.com/questions/39909655/listing-of-all-files-in-directory -
Getting all files in a dir
* https://stackoverflow.com/a/678242/8728749 - filepath stem
"""

import pathlib
from typing import Optional

from config import config


def get_filepath_for_problem(
    user_input: str, parent_path: Optional[str] = None
) -> Optional[str]:
    """
    Finds the full path for a given problem name, where the problem name is the TSPLIB
    file without an extension. For example, if a TSPLIB file exists at:
    config.problems_parent_path/lorem/ipsum/p25.tsp, the function would return this
    path when user_input=p25.

    Function works recursively such that it will search all subdirectories in the parent
    path for the file. If no file exists, returns None.

    Parameters
    ----------
    user_input: str
        Filename of problem without suffix, ex. berlin52 for berlin52.tsp
    parent_path: str
        Path to search for file under. Meant to be called with None so that
        config.problems_parent_path is used, and filled when called recursively.
        However, can also override from the start by manually specifying a parent path.

    Returns
    -------
    str or None
        Full path to the problem
    """
    if not parent_path:
        parent_path = config.problems_parent_path
    for file_obj in pathlib.Path(parent_path).iterdir():
        if file_obj.is_file():
            if file_obj.stem == user_input:
                return str(file_obj)
        else:
            if subdir_file_obj := get_filepath_for_problem(user_input, file_obj):
                return str(subdir_file_obj)
    return None


def get_available_problems(parent_path: Optional[str] = None) -> list[Optional[str]]:
    """
    Returns all problem names under the parent path. For example, if the following
    problems exist:
    * config.problems_parent_path/lorem/berlin52.tsp
    * config.problems_parent_path/ipsum/dolor/fri26.tsp
    The function will return [berlin52, fri26]

    Parameters
    ----------
    parent_path: str
        Path to search for problems under. Meant to be called with None so that
        config.problems_parent_path is used, and filled when called recursively.
        However, can also override from the start by manually specifying a parent path.

    Returns
    -------
    list
    """
    if not parent_path:
        parent_path = config.problems_parent_path
    problems = []
    for file_obj in pathlib.Path(parent_path).iterdir():
        if file_obj.is_file():
            if file_obj.suffix == config.problems_file_extension:
                problems.append(file_obj.stem)
        else:
            if subdir_problems := get_available_problems(file_obj):
                problems += subdir_problems
    return sorted(problems)
