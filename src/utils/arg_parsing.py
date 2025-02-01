from config import config


def get_filepath_for_problem(user_input):
    """
    Allows problems to be specified either by their order in the config.problems dict,
    the key of the dict, or their subpath, ex. papers/barachet10.tsp.

    Returns the full path, determining the parent path from the config.

    Workflow
    --------
    - Try to treat user_input as an int. If it as an int, treat it as an index and
    return the ith subpath, starting at index 0, in the config.problems dict with the
    parent path prepended.
    - If it is not an int, try to treat user_input as a key to the config.problems
    dictionary. Return the value specified by the key for dict with the parent dict
    prepended.
    - If it is not a key, treat the input as a subpath and return the parent path +
    user input.

    Parameters
    ----------
    user_input: str,int
        Specifying the problem as described in the Workflow

    Config Values
    -------------
    problems_parent_path - str
        String to prepend to the subpath in problem_dict
    problems - dict
        Key - Problem Name
        Value - Subpath

    Returns
    -------
    str
        Full path to problem
    """
    try:
        problem = int(user_input)
        subpath = list(config.problems.values())[problem]
    except ValueError:
        try:
            subpath = config.problems[user_input]
        except KeyError:
            subpath = user_input
    return f"{config.problems_parent_path}/{subpath}"
