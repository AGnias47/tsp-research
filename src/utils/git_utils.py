import git

REPO = git.Repo()


def get_commit() -> str:
    """
    Get commit of the current directory

    References
    ----------
    * https://stackoverflow.com/a/41210204/8728749 - Provided code for this function

    Returns
    -------
    str
    """
    return REPO.head.object.hexsha


def get_short_hash() -> True:
    """
    Get short hash of commit of current directory

    References
    ----------
    * https://stackoverflow.com/a/67470153/8728749 - Provided code

    Returns
    -------
    str
    """
    return REPO.git.rev_parse(REPO.head, short=True)
