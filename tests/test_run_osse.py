import pandas as pd
import pytest
from pyinverse.loss import Bayesian, BayesianYM

# flake8: noqa
from pyinverse.solver import BayesianAnalytical, BayesianAnalyticalYM

from flexwrfinversion.loaders.footprint import *
from flexwrfinversion.loaders.measurement import *
from flexwrfinversion.loaders.measurement_covariance import *
from flexwrfinversion.loaders.prior import *
from flexwrfinversion.loaders.prior_covariance import *

# flake8: noqa
from flexwrfinversion.loaders.target import *
from flexwrfinversion.run_osse import (
    _compute_inversion,
    _format_results,
    _get_kwargs,
    _initialize_loaders,
    _prepare_permutations,
    _select_sites,
)


def test_get_kwargs():
    config = {
        "prior": {
            "kwargs": {
                "a": 1,
                "b": 2,
            }
        },
        "prior_measurement": {
            "c": 3,
        },
        "prior_prior_covariance": {
            "d": 4,
        },
        "prior_covariance_measurement": {
            "e": 5,
        },
    }
    kwargs = _get_kwargs(config, "prior")
    assert kwargs == {"a": 1, "b": 2, "c": 3, "d": 4}


def test_initialize_loaders(
    example_config5,
):
    (
        target_loader,
        prior_loader,
        prior_covariance_loader,
        footprint_loader,
        measurement_loader,
        measurement_covariance_loader,
    ) = _initialize_loaders(example_config5)
    assert isinstance(target_loader, FlexibleTargetLoaderAnthBioCo)
    assert isinstance(prior_loader, FlatPrior)
    assert isinstance(prior_covariance_loader, TargetAsError)
    assert isinstance(footprint_loader, FlexibleFootprintLoaderAnthBioCo)
    assert isinstance(measurement_loader, FlexibleMeasurementLoaderTotalCo)
    assert isinstance(measurement_covariance_loader, ConstantNoCorrelationCO)
    assert footprint_loader._keep_only == [b"site00"]
    assert measurement_loader._keep_only == [b"site00"]


def test_select_sites(
    flexible_measurement_loader_total,
    flexible_footprint_loader_total,
):
    sites = np.array([b"site00"]).astype("|S6")
    convariance_loader = ConstantNoCorrelation(
        measurement_loader=flexible_measurement_loader_total, ppm_error=2
    )
    measurements = flexible_measurement_loader_total.measurements
    start_mtime = measurements.unstack().MTime.values[0]
    end_mtime = measurements.unstack().MTime.values[4]

    measurements, measurement_covariance, footprints = _select_sites(
        flexible_measurement_loader_total.measurements,
        convariance_loader.load_timeframe(start_mtime, end_mtime),
        flexible_footprint_loader_total.footprint,
        sites,
    )
    assert set(measurements.MPlace.values) == {b"site00"}
    assert set(measurement_covariance.MPlace0.values) == {b"site00"}
    assert set(measurement_covariance.MPlace1.values) == {b"site00"}
    assert set(footprints.MPlace.values) == {b"site00"}


def test_compute_inversion_bayesian():
    n_space = 3
    n_time = 2
    n_mspace = 4
    n_mtime = 5

    # Create synthetic data
    np.random.seed(0)
    prior = np.random.randn(n_time, n_space).astype(np.float32)
    prior_covariance = np.eye(n_space * n_time).astype(np.float32)
    footprint = np.random.randn(n_time, n_space, n_mspace * n_mtime).astype(np.float32)
    measurement = np.random.randn(n_mspace * n_mtime).astype(np.float32)
    measurement_covariance = np.eye(n_mspace * n_mtime).astype(np.float32)
    prior_spatial_correlation = np.eye(n_space).astype(np.float32)
    prior_temporal_correlation = np.eye(n_time).astype(np.float32)
    prior_std = np.ones_like(prior).astype(np.float32)
    stacked_prior = prior.flatten()
    stacked_footprint = footprint.reshape(n_space * n_time, n_mspace * n_mtime)

    # Create the Bayesian solver
    solver = _compute_inversion(
        Bayesian,
        BayesianAnalytical,
        x_prior=stacked_prior,
        cov_prior=prior_covariance,
        y=measurement,
        cov_y=measurement_covariance,
        K=stacked_footprint.T,
    )
    assert not solver._x_posterior is None
    assert not solver._cov_posterior is None
    post1 = solver.x_posterior

    # Create the BayesianYM solver
    solver = _compute_inversion(
        BayesianYM,
        BayesianAnalyticalYM,
        prior=prior,
        prior_standard_deviation=prior_std,
        prior_temporal_correlation=prior_temporal_correlation,
        prior_spatial_correlation=prior_spatial_correlation,
        forward_model=footprint.transpose(2, 0, 1),
        measurement=measurement,
        measurement_covariance=measurement_covariance,
    )

    assert not solver.solver._x_posterior is None
    assert not solver.solver._posterior_std is None
    post2, _ = solver()

    assert np.allclose(post1, post2.flatten(), rtol=1e-3, atol=0)


def test_format_results():
    start_date = np.datetime64("2020-01-01T00").astype("datetime64[ns]")
    end_date = np.datetime64("2020-01-02T00").astype("datetime64[ns]")
    n_time = 25
    n_space = 6

    class mock_solver:
        def __init__(self):
            self.x_posterior = np.random.randn(n_time * n_space)
            self.cov_posterior = np.eye(n_time * n_space)
            self.averaging_kernel = np.eye(n_time * n_space)

    state_dataarray = xr.DataArray(
        np.zeros(shape=(n_time, n_space)),
        dims=["Time", "Space"],
        coords={
            "Time": start_date
            + np.arange(n_time).astype("timedelta64[h]").astype("timedelta64[ns]"),
            "Space": np.arange(n_space),
        },
    )
    stacked_dataarray = state_dataarray.stack(state=("Time", "Space"))
    state_covariance = MeasurementCovarianceLoader._to_two_dimensions(state_dataarray)
    stacked_covariance = state_covariance.stack(
        state0=("Time0", "Space0"), state1=("Time1", "Space1")
    )
    (
        posterior_emissions,
        posterior_std,
        averaging_kernel_diag,
        averaging_kernel_sum,
    ) = _format_results(
        mock_solver(),
        stacked_dataarray.coords,
        stacked_covariance.coords,
        start_date,
        end_date,
        0,
    )
    assert isinstance(posterior_emissions, xr.DataArray)
    assert isinstance(posterior_std, xr.DataArray)
    assert isinstance(averaging_kernel_diag, xr.DataArray)
    assert isinstance(averaging_kernel_sum, xr.DataArray)

    assert set(posterior_emissions.dims) == {"Time", "Space"}
    assert set(posterior_std.dims) == {"Time", "Space"}
    assert set(averaging_kernel_diag.dims) == {"Time", "Space"}
    assert set(averaging_kernel_sum.dims) == {"Space0", "Space1", "buffer"}


def test_prepare_permutations(flexible_measurement_loader_total):
    config = {"n_permutations": 3, "n_stations": 1, "permutation_seed": 42}
    mplace_value_permutations = _prepare_permutations(
        config, flexible_measurement_loader_total
    )
    assert len(mplace_value_permutations) == 3
    assert len(mplace_value_permutations[0]) == 1


def test_prepare_permutations_with_co2_ff(measurement_loader_total_and_co2_ff):
    config = {"n_permutations": 3, "n_stations": 1, "permutation_seed": 42}
    mplace_value_permutations = _prepare_permutations(
        config, measurement_loader_total_and_co2_ff
    )
    assert len(mplace_value_permutations) == 3
    assert len(mplace_value_permutations[0]) == 2
    for permutation in mplace_value_permutations:
        assert (
            measurement_loader_total_and_co2_ff.CO2_FF_MPLACE_NAME.encode()
            in permutation
        )
        assert (
            permutation
            == measurement_loader_total_and_co2_ff.CO2_FF_MPLACE_NAME.encode()
        ).sum() <= 1
