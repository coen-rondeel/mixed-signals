"""Tests for the eryn adapters: spaces, physical transforms and prior densities."""

import math
import sys
import textwrap
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pydantic
import pytest

from mixed_signals.core.samples import SampleSet
from mixed_signals.core.space import ParamSpace
from mixed_signals.core.topology import PolarForm, Sphere, Topology
from mixed_signals.errors import MissingDependencyError
from mixed_signals.priors import ErynPhysical, ErynPrior, PriorDistribution
from mixed_signals.priors.eryn_container import PHYSICAL_DESCRIPTION
from mixed_signals.transforms import Compose, Whitening

eryn_priors = pytest.importorskip("eryn.priors")

TWO_PI = 2 * math.pi
FIXED_VALUE = 3.0
SAMPLE_COUNT = 64


def sky_priors() -> Any:
    """Return priors of a sky-located source, with every transform and a fixed value."""
    return eryn_priors.ProbDistContainer(
        {
            "ln_amplitude": eryn_priors.LogUniform(
                1e-23, 1e-21, name="ln_amplitude", name_phys="amplitude", unit="1"
            ),
            "phase": eryn_priors.UniformDistribution(0.0, TWO_PI),
            "longitude": eryn_priors.UniformDistribution(
                0.0, TWO_PI, name="longitude", latex_label=r"\lambda", unit="rad"
            ),
            "sin_latitude": eryn_priors.SineUniform(
                name="sin_latitude", name_phys="latitude"
            ),
            "cos_inclination": eryn_priors.CosineUniform(
                name="cos_inclination", name_phys="inclination", unit="1"
            ),
            "distance_cubed": eryn_priors.UniformInVolume(
                1.0, 10.0, name="distance_cubed", name_phys="distance"
            ),
            "fixed": eryn_priors.DeltaFunction(FIXED_VALUE),
        }
    )


SKY_PERIODIC = {"phase": TWO_PI, "longitude": TWO_PI}
SKY_SPHERE = Sphere(
    azimuth="longitude", polar="sin_latitude", polar_form=PolarForm.SIN_LATITUDE
)
FREE_NAMES = (
    "ln_amplitude",
    "phase",
    "longitude",
    "sin_latitude",
    "cos_inclination",
    "distance_cubed",
)


def sky_space() -> ParamSpace:
    """Return the sampling space of :func:`sky_priors`, with its sphere."""
    return ParamSpace.from_eryn_priors(
        sky_priors(), periodic=SKY_PERIODIC, spheres=[SKY_SPHERE]
    )


def free_draws(priors: Any, count: int = SAMPLE_COUNT) -> np.ndarray:
    """Return NumPy draws of the free parameters of the sky priors."""
    return np.asarray(ErynPrior(priors).rvs(count, key=jax.random.key(7)))


def with_fixed(free: np.ndarray) -> np.ndarray:
    """Return eryn's full vectors of the sky priors, with the fixed value last."""
    return np.concatenate([free, np.full((*free.shape[:-1], 1), FIXED_VALUE)], -1)


class CountPrior:
    """A discrete prior from outside eryn, recognised by its ``logpmf``."""

    def logpmf(self, values: np.ndarray) -> np.ndarray:
        """Return the log mass of a uniform count."""
        return np.zeros_like(values)

    def rvs(self, size: int = 1) -> np.ndarray:
        """Return zeros."""
        return np.zeros(size)


class PairPrior:
    """A prior with one unit over two parameters, as in eryn's conditional tests."""

    unit = "m"

    def logpdf(self, values: np.ndarray) -> np.ndarray:
        """Return a standard normal log density summed over the pair."""
        return -0.5 * np.sum(np.asarray(values) ** 2, axis=-1)

    def rvs(self, size: int = 1) -> np.ndarray:
        """Return zeros."""
        return np.zeros((size, 2))


# * Spaces


def test_names_labels_and_units_follow_the_priors() -> None:
    """Names are keys; labels and units are eryn's, with the name for unnamed priors."""
    space = sky_space()
    assert space.names == FREE_NAMES
    assert space["longitude"].label == r"\lambda"
    assert space["longitude"].unit == "rad"
    assert space["ln_amplitude"].label == "ln_amplitude"
    assert space["ln_amplitude"].unit == "1"
    # * eryn labels the unnamed phase prior "None"; the space uses the name.
    assert space["phase"].label == "phase"
    assert space["phase"].unit is None


def test_bounds_give_intervals_and_real_lines() -> None:
    """Sampling bounds become intervals, and priors without bounds real lines."""
    priors = eryn_priors.ProbDistContainer(
        {
            "ln_mass": eryn_priors.LogUniform(1.0, 100.0),
            "offset": eryn_priors.Gaussian(0.0, 1.0),
            "rate": eryn_priors.LogNormal(0.0, 1.0),
            "delay": eryn_priors.Exponential(2.0),
        }
    )
    space = ParamSpace.from_eryn_priors(priors)
    assert space["ln_mass"].topology is Topology.INTERVAL
    np.testing.assert_allclose(space["ln_mass"].bounds, (0.0, math.log(100.0)))
    assert space["offset"].topology is Topology.REAL
    assert space["rate"].bounds == (0.0, math.inf)
    assert space["delay"].topology is Topology.INTERVAL


def test_periodic_boundary_and_sampler_dict_give_circles() -> None:
    """A periodic boundary, a dict name or a dict position makes a circle."""
    priors = eryn_priors.ProbDistContainer(
        {
            "phase": eryn_priors.UniformDistribution(0.0, TWO_PI, boundary="periodic"),
            "polarisation": eryn_priors.UniformDistribution(0.0, math.pi),
            "longitude": eryn_priors.UniformDistribution(0.0, TWO_PI),
            "reflected": eryn_priors.UniformDistribution(
                0.0, 1.0, boundary="reflective"
            ),
        }
    )
    periods: dict[str | int, float] = {"polarisation": math.pi, 2: TWO_PI}
    space = ParamSpace.from_eryn_priors(priors, periodic=periods)
    for name, period in (
        ("phase", TWO_PI),
        ("polarisation", math.pi),
        ("longitude", TWO_PI),
    ):
        assert space[name].topology is Topology.CIRCLE
        assert space[name].period == pytest.approx(period)
    assert space["reflected"].topology is Topology.INTERVAL


def test_period_must_equal_the_prior_width() -> None:
    """A period other than the width of the prior raises."""
    priors = eryn_priors.ProbDistContainer(
        {"phase": eryn_priors.UniformDistribution(0.0, TWO_PI)}
    )
    with pytest.raises(ValueError, match="the period must equal that width"):
        ParamSpace.from_eryn_priors(priors, periodic={"phase": math.pi})


def test_periodic_axes_without_bounds_span_their_period() -> None:
    """Without finite bounds, a circle spans [0, P), where eryn wraps it."""
    priors = eryn_priors.ProbDistContainer(
        {"phase": eryn_priors.Gaussian(1.0, 0.5, boundary="periodic")}
    )
    space = ParamSpace.from_eryn_priors(priors, periodic={"phase": TWO_PI})
    assert space["phase"].topology is Topology.CIRCLE
    assert space["phase"].bounds == (0.0, TWO_PI)
    with pytest.raises(ValueError, match="give it in periodic="):
        ParamSpace.from_eryn_priors(priors)


@pytest.mark.parametrize(
    ("periodic", "message"),
    [
        ({"unknown": TWO_PI}, "not a parameter of the priors"),
        ({5: TWO_PI}, "names position 5"),
        ({"phase": -1.0}, "positive and finite"),
        ({"phase": math.inf}, "positive and finite"),
    ],
)
def test_periodic_dict_entries_are_checked(
    periodic: dict[str | int, float], message: str
) -> None:
    """Unknown names, positions outside the vector and bad periods raise."""
    priors = eryn_priors.ProbDistContainer(
        {"phase": eryn_priors.UniformDistribution(0.0, TWO_PI)}
    )
    with pytest.raises(ValueError, match=message):
        ParamSpace.from_eryn_priors(priors, periodic=periodic)


def test_fixed_parameters_are_left_out() -> None:
    """DeltaFunction parameters are not in the space."""
    assert "fixed" not in sky_space().names


def test_every_fixed_parameter_raises() -> None:
    """A container whose every parameter is fixed has no space."""
    priors = eryn_priors.ProbDistContainer({"fixed": eryn_priors.DeltaFunction(1.0)})
    with pytest.raises(ValueError, match="every parameter of the priors is fixed"):
        ParamSpace.from_eryn_priors(priors)


@pytest.mark.parametrize(
    "prior",
    [
        eryn_priors.DiscreteUniform(0, 3),
        eryn_priors.CategoricalDistribution(3),
        eryn_priors.PoissonDistribution(2.0),
        CountPrior(),
    ],
    ids=["DiscreteUniform", "CategoricalDistribution", "PoissonDistribution", "logpmf"],
)
def test_discrete_priors_raise(prior: Any) -> None:
    """The discrete classes of eryn and priors with logpmf raise, Poisson too."""
    priors = eryn_priors.ProbDistContainer(
        {"count": prior, "offset": eryn_priors.Gaussian(0.0, 1.0)}
    )
    with pytest.raises(ValueError, match="discrete prior"):
        ParamSpace.from_eryn_priors(priors)


def test_integer_keys_give_numbered_names() -> None:
    """Integer keys give x0, x1, ..., with eryn's labels."""
    joint = eryn_priors.MultivariateGaussian(
        ("first", "second"), np.zeros(2), np.eye(2)
    )
    priors = eryn_priors.ProbDistContainer(
        {
            0: eryn_priors.UniformDistribution(0.0, 1.0, name="fraction"),
            1: eryn_priors.LogUniform(1.0, 10.0),
            (2, 3): joint,
        }
    )
    space = ParamSpace.from_eryn_priors(priors)
    assert space.names == ("x0", "x1", "x2", "x3")
    assert [parameter.label for parameter in space.parameters] == [
        "fraction",
        "x1",
        "first",
        "second",
    ]


def test_joint_priors_give_real_lines_with_their_labels() -> None:
    """A JointPrior has no bounds per axis, so its axes are real lines."""
    joint = eryn_priors.MultivariateGaussian(
        ("m1", "m2"),
        np.array([30.0, 25.0]),
        np.array([[4.0, 1.0], [1.0, 3.0]]),
        latex_labels=("m_1", None),
        units=("Msun", "Msun"),
    )
    priors = eryn_priors.ProbDistContainer({("mass_1", "mass_2"): joint})
    space = ParamSpace.from_eryn_priors(priors)
    assert space.names == ("mass_1", "mass_2")
    assert [parameter.label for parameter in space.parameters] == ["m_1", "m2"]
    assert all(parameter.topology is Topology.REAL for parameter in space.parameters)
    assert space["mass_2"].unit == "Msun"


def test_periodic_joint_axes_need_the_sampler_period() -> None:
    """A periodic JointPrior axis takes [0, P) from the sampler dict."""
    joint = eryn_priors.JointPrior(("phase", "offset"), boundaries=("periodic", None))
    priors = eryn_priors.ProbDistContainer({("phase", "offset"): joint})
    space = ParamSpace.from_eryn_priors(priors, periodic={"phase": TWO_PI})
    assert space["phase"].bounds == (0.0, TWO_PI)
    assert space["offset"].topology is Topology.REAL


def test_nested_containers_are_read_recursively() -> None:
    """A nested container takes the outer key names and its own priors."""
    inner = eryn_priors.ProbDistContainer(
        {
            "a": eryn_priors.UniformDistribution(0.0, TWO_PI, boundary="periodic"),
            "b": eryn_priors.SineUniform(unit="1"),
        }
    )
    priors = eryn_priors.ProbDistContainer(
        {("longitude", "sin_latitude"): inner, "distance": eryn_priors.Gaussian(1, 1)}
    )
    space = ParamSpace.from_eryn_priors(priors)
    assert space.names == ("longitude", "sin_latitude", "distance")
    assert space["longitude"].topology is Topology.CIRCLE
    assert space["sin_latitude"].unit == "1"


def test_nested_containers_must_fit_their_key() -> None:
    """A nested container spans as many parameters as its key names."""
    inner = eryn_priors.ProbDistContainer(
        {"a": eryn_priors.Gaussian(0.0, 1.0), "b": eryn_priors.Gaussian(0.0, 1.0)}
    )
    priors = eryn_priors.ProbDistContainer({("x", "y", "z"): inner})
    with pytest.raises(ValueError, match="a nested container spans 2 parameters"):
        ParamSpace.from_eryn_priors(priors)


def test_joint_priors_must_fit_their_key() -> None:
    """A JointPrior spans as many parameters as its key names."""
    joint = eryn_priors.MultivariateGaussian(("a", "b"), np.zeros(2), np.eye(2))
    priors = eryn_priors.ProbDistContainer({("x", "y", "z"): joint})
    with pytest.raises(ValueError, match="spans 2 parameters"):
        ParamSpace.from_eryn_priors(priors)


def test_plain_priors_over_several_parameters_give_real_lines() -> None:
    """A one-label prior over several parameters gives real lines named by the key."""
    priors = eryn_priors.ProbDistContainer({("north", "east"): PairPrior()})
    space = ParamSpace.from_eryn_priors(priors)
    assert space.names == ("north", "east")
    assert all(parameter.topology is Topology.REAL for parameter in space.parameters)
    assert space["east"].unit is None


def test_overlapping_priors_raise() -> None:
    """A parameter under two keys has two priors, which the space rejects."""
    priors = eryn_priors.ProbDistContainer(
        {
            0: eryn_priors.Gaussian(0.0, 1.0),
            (0, 1): eryn_priors.MultivariateGaussian(
                ("a", "b"), np.zeros(2), np.eye(2)
            ),
        }
    )
    with pytest.raises(ValueError, match="'x0' has more than one prior"):
        ParamSpace.from_eryn_priors(priors)


def test_sphere_polar_forms_must_fit_their_priors() -> None:
    """SineUniform samples sin_latitude and CosineUniform cos_colatitude."""
    priors = eryn_priors.ProbDistContainer(
        {
            "longitude": eryn_priors.UniformDistribution(0.0, TWO_PI),
            "cos_colatitude": eryn_priors.CosineUniform(),
        }
    )
    periodic = {"longitude": TWO_PI}
    good = Sphere(
        azimuth="longitude", polar="cos_colatitude", polar_form="cos_colatitude"
    )
    space = ParamSpace.from_eryn_priors(priors, periodic=periodic, spheres=[good])
    assert space.spheres == (good,)
    bad = Sphere(azimuth="longitude", polar="cos_colatitude", polar_form="colatitude")
    with pytest.raises(ValueError, match="samples cos_colatitude"):
        ParamSpace.from_eryn_priors(priors, periodic=periodic, spheres=[bad])


def test_sphere_axes_must_not_be_fixed() -> None:
    """A sphere axis with a DeltaFunction prior raises."""
    priors = eryn_priors.ProbDistContainer(
        {
            "longitude": eryn_priors.UniformDistribution(0.0, TWO_PI),
            "latitude": eryn_priors.DeltaFunction(0.5),
        }
    )
    sphere = Sphere(azimuth="longitude", polar="latitude", polar_form="latitude")
    with pytest.raises(ValueError, match="'latitude' has a DeltaFunction prior"):
        ParamSpace.from_eryn_priors(
            priors, periodic={"longitude": TWO_PI}, spheres=[sphere]
        )


def test_spheres_with_unknown_names_fail_validation() -> None:
    """A sphere that names no parameter fails as any invalid space does."""
    priors = eryn_priors.ProbDistContainer(
        {"longitude": eryn_priors.UniformDistribution(0.0, TWO_PI)}
    )
    sphere = Sphere(azimuth="longitude", polar="missing", polar_form="latitude")
    with pytest.raises(pydantic.ValidationError, match="'missing' is not a parameter"):
        ParamSpace.from_eryn_priors(
            priors, periodic={"longitude": TWO_PI}, spheres=[sphere]
        )


def test_space_round_trips_through_yaml() -> None:
    """A space from eryn priors is an ordinary space."""
    space = sky_space()
    assert ParamSpace.from_yaml_text(space.to_yaml_text()) == space


def test_priors_must_be_a_container() -> None:
    """Plain dicts of priors raise; eryn wraps them in a ProbDistContainer."""
    with pytest.raises(TypeError, match="must be an eryn ProbDistContainer"):
        ParamSpace.from_eryn_priors({"phase": eryn_priors.Gaussian(0.0, 1.0)})


def test_missing_eryn_raises_with_the_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without eryn, the adapters name the extra that installs it."""
    monkeypatch.setitem(sys.modules, "eryn.priors", None)
    with pytest.raises(MissingDependencyError, match=r"mixed-signals\[eryn\]"):
        ParamSpace.from_eryn_priors(object())


# * Physical parameters


def test_physical_maps_are_eryns_transforms() -> None:
    """The forward map is eryn's inverse_transform, the inverse its forward one."""
    priors = sky_priors()
    physical = ErynPhysical(priors)
    sampling = free_draws(priors)
    expected = np.asarray(priors.inverse_transform(with_fixed(sampling)))[..., :-1]
    np.testing.assert_allclose(physical.forward(jnp.asarray(sampling)), expected)
    np.testing.assert_allclose(
        physical.inverse(jnp.asarray(expected)), sampling, rtol=1e-12, atol=1e-15
    )
    assert physical.free == tuple(range(6))
    assert physical.fixed == ((6, FIXED_VALUE),)
    # * The caller's container keeps its NumPy backend.
    assert priors.backend_name == "numpy"


def test_log_determinant_is_the_negative_eryn_jacobian() -> None:
    """ln|det ∂x/∂u| is −eryn's log_jacobian and matches the full Jacobian."""
    priors = sky_priors()
    physical = ErynPhysical(priors)
    sampling = jnp.asarray(free_draws(priors))
    expected = -np.asarray(
        priors.log_jacobian(priors.inverse_transform(with_fixed(np.asarray(sampling))))
    )
    np.testing.assert_allclose(physical.log_abs_det_jacobian(sampling), expected)
    _, log_determinant = jnp.linalg.slogdet(physical.jacobian(sampling))
    np.testing.assert_allclose(log_determinant, expected, rtol=1e-12)


def test_physical_transform_runs_under_jit_vmap_and_grad() -> None:
    """The maps trace, map over problems and differentiate."""
    priors = sky_priors()
    physical = ErynPhysical(priors)
    sampling = jnp.asarray(free_draws(priors, 4))
    np.testing.assert_allclose(
        jax.jit(physical.forward)(sampling), physical.forward(sampling)
    )
    batched = jax.vmap(physical.forward)(sampling[None])
    np.testing.assert_allclose(batched[0], physical.forward(sampling))
    gradient = jax.grad(lambda point: physical.log_abs_det_jacobian(point))(sampling[0])
    assert bool(jnp.all(jnp.isfinite(gradient)))


def test_fisher_matrices_map_with_eryns_jacobians() -> None:
    """A unit Fisher matrix of u maps to diag((∂u/∂x)²) of the physical parameters."""
    priors = sky_priors()
    physical = ErynPhysical(priors)
    sampling = jnp.asarray(free_draws(priors, 3))
    physical_points = np.asarray(physical.forward(sampling))
    amplitude, latitude, inclination, distance = (
        physical_points[:, index] for index in (0, 3, 4, 5)
    )
    derivatives = np.stack(
        [
            1 / amplitude,
            np.ones_like(amplitude),
            np.ones_like(amplitude),
            np.cos(latitude),
            -np.sin(inclination),
            3 * distance**2,
        ],
        axis=-1,
    )
    fisher = jnp.broadcast_to(jnp.eye(6), (3, 6, 6))
    mapped = physical.map_fisher(fisher, sampling)
    np.testing.assert_allclose(
        mapped,
        np.einsum("ni,ij->nij", derivatives**2, np.eye(6)),
        rtol=1e-10,
        atol=1e-12,
    )


def test_sample_sets_carry_the_physical_prior_density() -> None:
    """Mapped log_prior equals eryn's log prior density of the physical parameters."""
    priors = sky_priors()
    physical = ErynPhysical(priors)
    sampling = free_draws(priors)
    sample_set = SampleSet(
        sampling, log_prior=np.asarray(priors.logpdf(with_fixed(sampling)))
    )
    mapped = physical.map_sample_set(sample_set)
    physical_points = with_fixed(np.asarray(mapped.samples[0, 0]))
    assert mapped.log_prior is not None
    np.testing.assert_allclose(
        mapped.log_prior[0, 0], priors.logpdf_physical(physical_points), rtol=1e-12
    )


def test_description_names_the_physical_space() -> None:
    """The space column reads physical(eryn), also inside a composition."""
    priors = sky_priors()
    physical = ErynPhysical(priors)
    assert physical.description == PHYSICAL_DESCRIPTION == "physical(eryn)"
    physical_space = physical.map_space(sky_space())
    samples = physical.forward(jnp.asarray(free_draws(priors)))
    whitening = Whitening.from_samples(samples[None], None, physical_space)
    assert (
        Compose((physical, whitening)).description
        == f"physical(eryn) | {whitening.description}"
    )


def test_map_space_renames_transformed_axes() -> None:
    """Transformed axes take name_phys, as name and label, without a unit."""
    physical_space = ErynPhysical(sky_priors()).map_space(sky_space())
    assert physical_space.names == (
        "amplitude",
        "phase",
        "longitude",
        "latitude",
        "inclination",
        "distance",
    )
    assert physical_space["amplitude"].label == "amplitude"
    assert physical_space["amplitude"].unit is None
    assert physical_space["inclination"].unit is None
    # * The untransformed longitude keeps its label and unit.
    assert physical_space["longitude"].label == r"\lambda"
    assert physical_space["longitude"].unit == "rad"
    assert physical_space["phase"].topology is Topology.CIRCLE


def test_map_space_maps_bounds_in_increasing_order() -> None:
    """Bounds are the inverse transforms of the sampling bounds, sorted."""
    physical_space = ErynPhysical(sky_priors()).map_space(sky_space())
    np.testing.assert_allclose(physical_space["amplitude"].bounds, (1e-23, 1e-21))
    np.testing.assert_allclose(physical_space["distance"].bounds, (1.0, 10.0))
    # * arccos reverses the order of the cos bounds.
    lower, upper = physical_space["inclination"].bounds
    assert lower < upper
    np.testing.assert_allclose(
        (lower, upper), (math.acos(1 - 1e-12), math.acos(-1 + 1e-12))
    )


def test_map_space_maps_sphere_polar_forms() -> None:
    """sin_latitude becomes latitude, and cos_colatitude colatitude."""
    physical_space = ErynPhysical(sky_priors()).map_space(sky_space())
    assert physical_space.spheres == (
        Sphere(azimuth="longitude", polar="latitude", polar_form="latitude"),
    )
    priors = eryn_priors.ProbDistContainer(
        {
            "longitude": eryn_priors.UniformDistribution(0.0, TWO_PI),
            "cos_colatitude": eryn_priors.CosineUniform(name_phys="colatitude"),
        }
    )
    sphere = Sphere(
        azimuth="longitude", polar="cos_colatitude", polar_form="cos_colatitude"
    )
    space = ParamSpace.from_eryn_priors(
        priors, periodic={"longitude": TWO_PI}, spheres=[sphere]
    )
    mapped = ErynPhysical(priors).map_space(space)
    assert mapped.spheres[0].polar_form is PolarForm.COLATITUDE
    assert mapped.spheres[0].polar == "colatitude"


def test_map_space_keeps_spheres_without_transforms() -> None:
    """A sphere sampled in latitude itself stays as it is."""
    priors = eryn_priors.ProbDistContainer(
        {
            "longitude": eryn_priors.UniformDistribution(0.0, TWO_PI),
            "latitude": eryn_priors.UniformDistribution(-math.pi / 2, math.pi / 2),
            "ln_distance": eryn_priors.LogUniform(1.0, 10.0, name_phys="distance"),
        }
    )
    sphere = Sphere(azimuth="longitude", polar="latitude", polar_form="latitude")
    space = ParamSpace.from_eryn_priors(
        priors, periodic={"longitude": TWO_PI}, spheres=[sphere]
    )
    mapped = ErynPhysical(priors).map_space(space)
    assert mapped.spheres == (sphere,)
    assert mapped.names == ("longitude", "latitude", "distance")


def test_map_space_rejects_transformed_circles() -> None:
    """A transformed coordinate has no constant period."""
    priors = eryn_priors.ProbDistContainer({"ln_period": eryn_priors.LogUniform(1, 10)})
    width = math.log(10.0)
    space = ParamSpace.from_eryn_priors(priors, periodic={"ln_period": width})
    with pytest.raises(ValueError, match="is a circle with a LogUniform transform"):
        ErynPhysical(priors).map_space(space)


def test_map_space_rejects_other_transforms_on_spheres() -> None:
    """Only the sine and cosine priors map the polar axis of a sphere."""
    priors = eryn_priors.ProbDistContainer(
        {
            "longitude": eryn_priors.UniformDistribution(0.0, TWO_PI),
            "height": eryn_priors.PowerLaw(2.0, 0.0, 1.0),
        }
    )
    sphere = Sphere(azimuth="longitude", polar="height", polar_form="sin_latitude")
    space = ParamSpace.from_eryn_priors(
        priors, periodic={"longitude": TWO_PI}, spheres=[sphere]
    )
    with pytest.raises(ValueError, match="only SineUniform and CosineUniform"):
        ErynPhysical(priors).map_space(space)


def test_map_space_checks_the_polar_form_of_hand_made_spaces() -> None:
    """A space that gives a SineUniform axis another form raises."""
    priors = eryn_priors.ProbDistContainer(
        {
            "longitude": eryn_priors.UniformDistribution(0.0, TWO_PI),
            "polar": eryn_priors.SineUniform(),
        }
    )
    space = ParamSpace.from_yaml_text(
        textwrap.dedent(
            """
            parameters:
              - {name: longitude, topology: circle, bounds: [0.0, 6.283185307179586]}
              - {name: polar, topology: interval, bounds: [-0.5, 0.5]}
            spheres:
              - {azimuth: longitude, polar: polar, polar_form: latitude}
            """
        )
    )
    with pytest.raises(ValueError, match="samples sin_latitude"):
        ErynPhysical(priors).map_space(space)


def test_map_space_needs_one_parameter_per_free_position() -> None:
    """A space of another size raises."""
    space = ParamSpace.from_yaml_text("parameters:\n  - name: alone\n")
    with pytest.raises(ValueError, match="the space has 1 parameters"):
        ErynPhysical(sky_priors()).map_space(space)


def test_map_space_rejects_nan_bounds() -> None:
    """A transform whose inverse is NaN at a bound cannot map the space."""
    square = eryn_priors.UniformDistribution(
        -1.0,
        1.0,
        forward_transform=lambda values, **kwargs: values**2,
        inverse_transform=lambda values, **kwargs: jnp.sqrt(values),
        log_jacobian=lambda values, **kwargs: jnp.log(2 * jnp.abs(values)),
    )
    priors = eryn_priors.ProbDistContainer({"square": square})
    space = ParamSpace.from_eryn_priors(priors)
    with pytest.raises(ValueError, match="is NaN"):
        ErynPhysical(priors).map_space(space)


def test_points_must_hold_the_free_parameters() -> None:
    """Points with the fixed parameter, or any other count, raise."""
    priors = sky_priors()
    with pytest.raises(ValueError, match="points hold 7 parameters"):
        ErynPhysical(priors).forward(jnp.asarray(with_fixed(free_draws(priors))))


def test_eryn_without_a_jax_backend_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """An eryn whose priors have no JAX backend cannot give the transform."""
    priors = sky_priors()
    monkeypatch.delattr(eryn_priors.ProbDistContainer, "to_backend")
    with pytest.raises(MissingDependencyError, match="no JAX backend"):
        ErynPhysical(priors)


# * Prior densities


def test_prior_logpdf_is_eryns_logpdf() -> None:
    """The log density is eryn's at the full vectors, for any leading shape."""
    priors = sky_priors()
    prior = ErynPrior(priors)
    sampling = free_draws(priors).reshape(8, 8, 6)
    expected = np.asarray(priors.logpdf(with_fixed(sampling).reshape(-1, 7)))
    np.testing.assert_allclose(prior.logpdf(sampling), expected.reshape(8, 8))
    np.testing.assert_allclose(jax.jit(prior.logpdf)(sampling[0]), expected[:8])


def test_prior_draws_follow_the_key() -> None:
    """The same key gives the same draws, inside the priors' support."""
    prior = ErynPrior(sky_priors())
    draws = prior.rvs((2, 50), key=jax.random.key(3))
    assert draws.shape == (2, 50, 6)
    np.testing.assert_array_equal(draws, prior.rvs((2, 50), key=jax.random.key(3)))
    assert not np.array_equal(draws, prior.rvs((2, 50), key=jax.random.key(4)))
    assert bool(jnp.all(jnp.isfinite(prior.logpdf(draws))))


def test_prior_meets_the_protocol() -> None:
    """ErynPrior is a PriorDistribution."""
    prior: PriorDistribution = ErynPrior(sky_priors())
    assert isinstance(prior, PriorDistribution)
    assert prior.rvs(1, key=jax.random.key(0)).shape == (1, 6)
