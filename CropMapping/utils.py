from typing import Any


# function to return key for any value
def get_key(v: Any, dict_: dict) -> Any:
    for key, value in dict_.items():
        if value == v:
            return key
