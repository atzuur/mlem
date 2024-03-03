from typing import Any, Iterable

def row_of_doa(data: dict[Any, Iterable], idx: int) -> dict:
    """ get individual row from dict of arrays """
    return {key: col[idx] for key, col in data.items()}
