"""Tests for the loaders of the gw extra and the redshift inversion."""

import importlib
import math
import subprocess
import sys
import warnings
from collections.abc import Callable

import astropy.units as astropy_units
import numpy as np
import pytest
from astropy.cosmology import Planck15

from mixed_signals.errors import MissingDependencyError
from mixed_signals.gw.lisa import (
    GalacticBinaries,
    MassiveBlackHoleBinaries,
    _cosmology,
    _dependencies,
)
from mixed_signals.gw.lisa._cosmology import redshift_from_luminosity_distance
from mixed_signals.gw.lisa._dependencies import astropy_cosmology, lisa_constants

type Builder = Callable[..., dict[str, np.ndarray]]


def test_the_constants_come_from_lisaconstants() -> None:
    """The obliquity is in radians, and the other constants are in SI units."""
    import lisaconstants

    constants = lisa_constants()
    assert constants.obliquity == pytest.approx(
        math.radians(lisaconstants.OBLIQUITY), rel=1e-15
    )
    assert constants.obliquity == pytest.approx(0.409_092_600_600_582_9, rel=1e-12)
    assert constants.speed_of_light == lisaconstants.SPEED_OF_LIGHT == 299_792_458.0
    assert constants.gravitational_constant == lisaconstants.GRAVITATIONAL_CONSTANT
    assert constants.solar_mass == lisaconstants.SOLAR_MASS
    assert constants.parsec == lisaconstants.PARSEC
    assert astropy_cosmology().Planck15 is Planck15


def test_a_missing_package_raises_an_error_that_names_the_extra(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When an import fails, the error names the gw extra and the package."""

    def failing(name: str) -> object:
        raise ImportError(f"No module named {name!r}")

    monkeypatch.setattr(importlib, "import_module", failing)
    with pytest.raises(
        MissingDependencyError,
        match=r"cannot import lisaconstants.*mixed-signals\[gw\]",
    ):
        lisa_constants()
    with pytest.raises(
        MissingDependencyError,
        match=r"cannot import astropy.cosmology.*mixed-signals\[gw\]",
    ):
        astropy_cosmology()
    _cosmology._grid.cache_clear()
    with pytest.raises(MissingDependencyError, match="gw extra"):
        redshift_from_luminosity_distance(np.array([1.0]))


def test_only_calls_that_need_a_dependency_raise(
    monkeypatch: pytest.MonkeyPatch, lisa_posteriors: Builder
) -> None:
    """Holders build without the extra; the redshift and the sky rotation need it."""
    sources = MassiveBlackHoleBinaries.from_posteriors(
        lisa_posteriors("massive_black_hole_binary", {"a": 8}),
        name="fit",
        epoch=0.0,
        source_frame="J",
    )
    _cosmology._grid.cache_clear()
    for module in ("astropy", "astropy.cosmology", "lisaconstants"):
        monkeypatch.setitem(sys.modules, module, None)
    assert sources.total_mass.shape == (1, 8)
    with pytest.raises(MissingDependencyError, match="gw extra"):
        sources.redshift  # noqa: B018
    with pytest.raises(MissingDependencyError, match="gw extra"):
        sources.to_ecliptic()
    binaries = GalacticBinaries.from_posteriors(
        lisa_posteriors("galactic_binary", {"a": 8}), name="fit", epoch=0.0
    )
    with pytest.raises(MissingDependencyError, match="gw extra"):
        binaries.chirp_mass  # noqa: B018


def test_the_package_imports_without_the_extra() -> None:
    """A fresh interpreter without astropy and lisaconstants imports gw.lisa."""
    code = (
        "import sys\n"
        "sys.modules['astropy'] = None\n"
        "sys.modules['lisaconstants'] = None\n"
        "import mixed_signals.gw.lisa as lisa\n"
        "assert lisa.GalacticBinaries.source_type == 'galactic_binary'\n"
    )
    completed = subprocess.run(
        [sys.executable, "-W", "ignore::DeprecationWarning", "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


def test_the_lisaconstants_import_warning_is_filtered() -> None:
    """The loader imports lisaconstants without the warning that astropy 8 triggers."""
    code = (
        "import warnings\n"
        "warnings.simplefilter('error', UserWarning)\n"
        "from mixed_signals.gw.lisa._dependencies import lisa_constants\n"
        "print(lisa_constants().obliquity)\n"
    )
    completed = subprocess.run(
        [sys.executable, "-W", "ignore::DeprecationWarning", "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.startswith("0.409")
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        _dependencies._import("lisaconstants")


def test_the_redshift_inversion_is_accurate_and_monotone() -> None:
    """D_L(z(D)) returns D to 10⁻¹¹, and the map is increasing."""
    distances = np.geomspace(1e-4, 1e4, 300)
    redshift = redshift_from_luminosity_distance(distances)
    back = Planck15.luminosity_distance(redshift).to_value(astropy_units.Gpc)
    np.testing.assert_allclose(back, distances, rtol=1e-11)
    assert (np.diff(redshift) > 0).all()
    assert redshift_from_luminosity_distance(np.array(2.0)).shape == ()
    assert redshift_from_luminosity_distance(np.ones((3, 4))).shape == (3, 4)


def test_the_redshift_inversion_refuses_distances_it_cannot_invert() -> None:
    """Distances must be positive, finite and inside the tabulated range."""
    for bad in (-1.0, 0.0, math.nan, math.inf):
        with pytest.raises(ValueError, match="positive and finite"):
            redshift_from_luminosity_distance(np.array([1.0, bad]))
    with pytest.raises(ValueError, match="must lie between"):
        redshift_from_luminosity_distance(np.array([1e-12]))
    with pytest.raises(ValueError, match="must lie between"):
        redshift_from_luminosity_distance(np.array([1e12]))
