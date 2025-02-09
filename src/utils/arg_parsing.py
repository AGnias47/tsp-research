"""
References
----------
* https://stackoverflow.com/questions/39909655/listing-of-all-files-in-directory -
Getting all files in a dir
* https://stackoverflow.com/a/678242/8728749 - filepath stem
"""

import pathlib

from config import config


def get_filepath_for_problem(user_input, parent_path=None):
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


def get_available_problems(parent_path=None):
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
