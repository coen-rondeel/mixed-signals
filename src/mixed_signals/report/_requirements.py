"""Check that the optional ``report`` extra is installed, when the report is imported.

The report needs marimo, which runs the notebook, and cantuccio, which draws the
corner plots and brings matplotlib. Both belong to the ``report`` extra, so the
package imports without them and only :mod:`mixed_signals.report` needs them.
Importing this module checks both and raises
:class:`~mixed_signals.errors.MissingDependencyError`, whose message names the
extra, as :mod:`mixed_signals.io.hdf5` does for h5py.
"""

import importlib
from types import ModuleType
from typing import Final

from mixed_signals.errors import MissingDependencyError

REQUIRED_MODULES: Final = ("marimo", "cantuccio")
"""The modules of the ``report`` extra that the report imports."""

_INSTALL_HINT: Final = (
    'install it with the report extra: pip install "mixed-signals[report]"'
)


def require_module(name: str) -> ModuleType:
    """Import a module of the ``report`` extra.

    Args:
        name: The name of the module.

    Returns:
        The imported module.

    Raises:
        MissingDependencyError: If the module is not installed. The message names
            the extra to install.
    """
    try:
        return importlib.import_module(name)
    except ImportError as error:
        raise MissingDependencyError(
            f"cannot import {name} ({error}); {_INSTALL_HINT}"
        ) from error


for _module_name in REQUIRED_MODULES:
    require_module(_module_name)
