import functools
import warnings
from typing import Callable, Optional

try:
    from numba import njit, typed, types

except ImportError:

    warnings.warn(
        "numba not installed, falling back to stubs. "
        "Install numba for better napari-graph performance."
    )

    def njit(func: Optional[Callable] = None, **kwargs) -> Callable:
        """Immitate numba.njit decorator"""

        def _decorator(f: Callable) -> Callable:
            @functools.wraps(f)
            def wrapper(*args, **kwargs):
                return f(*args, **kwargs)

            return wrapper

        if func:
            return _decorator(func)

        return _decorator

    class StubList(list):
        @staticmethod
        def empty_list(type) -> list:
            return []

    class StubDict(dict):
        @staticmethod
        def empty(key_type, value_type) -> dict:
            return {}

    class typed:  # type: ignore[no-redef]
        List = StubList
        Dict = StubDict

    class types:  # type: ignore[no-redef]
        int64 = int
