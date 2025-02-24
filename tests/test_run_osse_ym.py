from unittest.mock import patch

import numpy as np
import pandas as pd
import xarray as xr

from flexwrfinversion.loaders.measurement_bias import ConstantBias
from flexwrfinversion.loaders.measurement_covariance import ConstantNoCorrelation
from flexwrfinversion.run_osse_ym import (
    _initialize_loaders,
    _load_inversion_data,
    _restack_coords,
    _run_inversion_ym_with_globals,
    _setup_restacking,
)


def test_run_inversion_ym_with_globals(tmp_path, flexible_measurement_loader_total):
    measurement_covariance_loader = ConstantNoCorrelation(
        flexible_measurement_loader_total, ppm_error=2
    )
    # Create synthetic data
    mplace = np.char.encode(np.array(["site1", "site2", "site3"]))
    mtime = pd.date_range("2020-01-02", periods=10, freq="h").to_numpy()
    time = pd.date_range("2020-01-01T22", periods=12, freq="h").to_numpy()
    subsectors = np.arange(5)
    sites = mplace[:2]
    prior_emissions = xr.DataArray(
        np.random.randn(12, 5), coords={"Time": time, "subsector": subsectors}
    )
    prior_standard_deviation = xr.DataArray(
        np.ones((12, 5)), coords={"Time": time, "subsector": subsectors}
    )
    prior_temporal_correlation = xr.DataArray(
        np.eye(12), coords={"Time0": time, "Time1": time}
    )
    prior_spatial_correlation = xr.DataArray(
        np.eye(5), coords={"subsector0": subsectors, "subsector1": subsectors}
    )
    footprints = (
        xr.DataArray(
            np.ones((3, 10, 12, 5)),
            coords={
                "MPlace": mplace,
                "MTime": mtime,
                "Time": time,
                "subsector": subsectors,
            },
        )
        .stack(measurement=["MPlace", "MTime"])
        .transpose("measurement", "Time", "subsector")
    )
    measurements = xr.DataArray(
        np.random.randn(3, 10), coords={"MPlace": mplace, "MTime": mtime}
    ).stack(measurement=["MPlace", "MTime"])
    measurement_covariance = xr.DataArray(
        np.eye(3)[:, None, :, None] * np.eye(10)[None, :, None, :],
        coords={"MPlace0": mplace, "MTime0": mtime, "MPlace1": mplace, "MTime1": mtime},
    ).stack(measurement0=["MPlace0", "MTime0"], measurement1=["MPlace1", "MTime1"])
    target_emissions = xr.DataArray(
        np.random.randn(12, 5), coords={"Time": time, "subsector": subsectors}
    )

    old_posterior = None
    for measurement_bias_loader in [
        None,
        ConstantBias(
            flexible_measurement_loader_total, measurement_covariance_loader, bias=3
        ),
    ]:
        with patch.multiple(
            "flexwrfinversion.run_osse_ym",
            prior_emissions=prior_emissions.astype(np.float32),
            prior_standard_deviation=prior_standard_deviation.astype(np.float32),
            prior_temporal_correlation=prior_temporal_correlation.astype(np.float32),
            prior_spatial_correlation=prior_spatial_correlation.astype(np.float32),
            footprints=footprints.astype(np.float32),
            measurements=measurements.astype(np.float32),
            measurement_covariance=measurement_covariance.astype(np.float32),
            target_emissions=target_emissions.astype(np.float32),
            mplace_value_permutations=[sites],
            output_buffer_path=tmp_path,
            measurement_bias_loader=measurement_bias_loader,
        ):
            _ = _run_inversion_ym_with_globals(0)

        result = xr.load_dataset(tmp_path / "permutation_0.nc")
        assert isinstance(result, xr.Dataset)
        assert "posterior_emissions" in result
        assert "posterior_std" in result

        posterior_emissions = result.posterior_emissions
        posterior_std = result.posterior_std

        assert (
            set(posterior_emissions.dims) - {"permutation"}
            == set(prior_emissions.dims)
            == {"Time", "subsector"}
        )
        assert (
            set(posterior_std.dims) - {"permutation"}
            == set(prior_standard_deviation.dims)
            == {"Time", "subsector"}
        )
        for dim in prior_emissions.dims:
            assert (result.posterior_emissions[dim] == prior_emissions[dim]).all()
            assert (result.posterior_std[dim] == prior_standard_deviation[dim]).all()

        if old_posterior is not None:
            assert not (posterior_emissions == old_posterior).all()
            assert not (posterior_std == old_posterior).all()
        old_posterior = posterior_emissions


def test_run_inversion_ym_with_globals_timeslices(
    tmp_path, flexible_measurement_loader_total
):
    n_times = 102
    len_footprint = 2
    n_slices = 6
    n_buffer_steps = 6
    n_meas_buffer_steps = 2

    n_subsectors = 1
    # Create synthetic data
    mplace = np.char.encode(np.array(["site1", "site2", "site3"]))
    mtime = pd.date_range(
        "2020-01-02", periods=n_times - len_footprint, freq="h"
    ).to_numpy()
    time = pd.date_range("2020-01-01T22", periods=n_times, freq="h").to_numpy()
    subsectors = np.arange(n_subsectors)
    sites = mplace[:1]
    prior_emissions = xr.DataArray(
        np.zeros(shape=(n_times, n_subsectors)),
        coords={"Time": time, "subsector": subsectors},
    )
    prior_standard_deviation = xr.DataArray(
        np.ones((n_times, n_subsectors)), coords={"Time": time, "subsector": subsectors}
    )
    prior_temporal_correlation = xr.DataArray(
        np.eye(n_times), coords={"Time0": time, "Time1": time}
    )
    prior_spatial_correlation = xr.DataArray(
        np.eye(n_subsectors),
        coords={"subsector0": subsectors, "subsector1": subsectors},
    )
    footprints = (
        xr.DataArray(
            np.ones((3, n_times - len_footprint, n_times, n_subsectors)),
            coords={
                "MPlace": mplace,
                "MTime": mtime,
                "Time": time,
                "subsector": subsectors,
            },
        )
        .stack(measurement=["MPlace", "MTime"])
        .transpose("measurement", "Time", "subsector")
    )
    mask = (time[:, None] <= mtime[None, :]) & (
        time[:, None] >= mtime[None, :] - np.timedelta64(len_footprint, "h")
    )
    footprints = (
        (footprints.unstack() * mask[:, None, None, :])
        .stack(measurement=["MPlace", "MTime"])
        .transpose("measurement", "Time", "subsector")
    )

    measurements = xr.DataArray(
        20 * np.ones(shape=(3, n_times - len_footprint)),
        coords={"MPlace": mplace, "MTime": mtime},
    ).stack(measurement=["MPlace", "MTime"])
    measurement_covariance = xr.DataArray(
        np.eye(3)[:, None, :, None] * np.eye(n_times - len_footprint)[None, :, None, :],
        coords={"MPlace0": mplace, "MTime0": mtime, "MPlace1": mplace, "MTime1": mtime},
    ).stack(measurement0=["MPlace0", "MTime0"], measurement1=["MPlace1", "MTime1"])
    target_emissions = xr.DataArray(
        np.random.randn(n_times, n_subsectors),
        coords={"Time": time, "subsector": subsectors},
    )

    original_buffer = tmp_path / "original"
    new_buffer = tmp_path / "new"
    original_buffer.mkdir()
    new_buffer.mkdir()

    with patch.multiple(
        "flexwrfinversion.run_osse_ym",
        prior_emissions=prior_emissions.astype(np.float32),
        prior_standard_deviation=prior_standard_deviation.astype(np.float32),
        prior_temporal_correlation=prior_temporal_correlation.astype(np.float32),
        prior_spatial_correlation=prior_spatial_correlation.astype(np.float32),
        footprints=footprints.astype(np.float32),
        measurements=measurements.astype(np.float32),
        measurement_covariance=measurement_covariance.astype(np.float32),
        target_emissions=target_emissions.astype(np.float32),
        mplace_value_permutations=[sites],
        output_buffer_path=original_buffer,
        measurement_bias_loader=None,
        n_temporal_slices=None,
        temporal_buffer_steps=0,
        meas_buffer_steps=0,
    ):
        _ = _run_inversion_ym_with_globals(0)

    with patch.multiple(
        "flexwrfinversion.run_osse_ym",
        prior_emissions=prior_emissions.astype(np.float32),
        prior_standard_deviation=prior_standard_deviation.astype(np.float32),
        prior_temporal_correlation=prior_temporal_correlation.astype(np.float32),
        prior_spatial_correlation=prior_spatial_correlation.astype(np.float32),
        footprints=footprints.astype(np.float32),
        measurements=measurements.astype(np.float32),
        measurement_covariance=measurement_covariance.astype(np.float32),
        target_emissions=target_emissions.astype(np.float32),
        mplace_value_permutations=[sites],
        output_buffer_path=new_buffer,
        measurement_bias_loader=None,
        n_temporal_slices=n_slices,
        temporal_buffer_steps=n_buffer_steps,
        meas_buffer_steps=n_meas_buffer_steps,
    ):
        _ = _run_inversion_ym_with_globals(0)

    original_result = xr.load_dataset(original_buffer / "permutation_0.nc")
    new_result = xr.load_dataset(new_buffer / "permutation_0.nc")
    assert isinstance(new_result, xr.Dataset)
    assert "posterior_emissions" in new_result
    assert "posterior_std" in new_result

    posterior_emissions = new_result.posterior_emissions
    posterior_std = new_result.posterior_std

    assert (
        set(posterior_emissions.dims) - {"permutation"}
        == set(prior_emissions.dims)
        == {"Time", "subsector"}
    )
    assert (
        set(posterior_std.dims) - {"permutation"}
        == set(prior_standard_deviation.dims)
        == {"Time", "subsector"}
    )
    for dim in prior_emissions.dims:
        assert (new_result.posterior_emissions[dim] == prior_emissions[dim]).all()
        assert (new_result.posterior_std[dim] == prior_standard_deviation[dim]).all()

    assert np.allclose(
        posterior_emissions, original_result.posterior_emissions, atol=0, rtol=1e-2
    )
    assert np.allclose(posterior_std, original_result.posterior_std, atol=0, rtol=1e-2)


def test_setup_restacking():
    class DummyPriorLoader:
        prior = xr.DataArray(np.random.randn(10, 5), dims=["Time", "subsector"])

    (
        dims_to_stack,
        state_order,
        footprint_order,
        prior_spatial_correlation_order,
        prior_spatial_correlation_dims_to_stack,
    ) = _setup_restacking(DummyPriorLoader())

    assert dims_to_stack == {}
    assert state_order == ["Time", "subsector"]
    assert footprint_order == ["measurement", "Time", "subsector"]
    assert prior_spatial_correlation_order == ["subsector0", "subsector1"]
    assert prior_spatial_correlation_dims_to_stack == {}


def test_setup_restacking_with_dims_to_stack():
    class DummyPriorLoader:
        prior = xr.DataArray(
            np.random.randn(10, 5, 3), dims=["Time", "subsector", "sector"]
        ).stack(state=["Time", "subsector", "sector"])

    (
        dims_to_stack,
        state_order,
        footprint_order,
        prior_spatial_correlation_order,
        prior_spatial_correlation_dims_to_stack,
    ) = _setup_restacking(DummyPriorLoader())
    assert dims_to_stack == {"subsector_sector": ["subsector", "sector"]}
    assert state_order == ["Time", "subsector_sector"]
    assert footprint_order == ["measurement", "Time", "subsector_sector"]
    assert prior_spatial_correlation_order == [
        "subsector0_sector0",
        "subsector1_sector1",
    ]
    assert prior_spatial_correlation_dims_to_stack == {
        "subsector0_sector0": ["subsector0", "sector0"],
        "subsector1_sector1": ["subsector1", "sector1"],
    }


def test_load_inversion_data(example_config5):
    (
        target_loader,
        prior_loader,
        prior_covariance_loader,
        footprint_loader,
        measurement_loader,
        measurement_covariance_loader,
    ) = _initialize_loaders(example_config5)

    (
        prior_emissions,
        prior_standard_deviation,
        prior_temporal_correlation,
        prior_spatial_correlation,
        footprints,
        target_emissions,
        measurements,
        measurement_covariance,
    ) = _load_inversion_data(
        prior_loader=prior_loader,
        prior_covariance_loader=prior_covariance_loader,
        footprint_loader=footprint_loader,
        measurement_loader=measurement_loader,
        measurement_covariance_loader=measurement_covariance_loader,
        target_loader=target_loader,
    )
    assert set(prior_emissions.dims) == {"state"}
    assert prior_emissions.dtype == np.float32
    assert set(prior_standard_deviation.dims) == {"state"}
    assert prior_standard_deviation.dtype == np.float32
    assert set(prior_temporal_correlation.dims) == {"Time0", "Time1"}
    assert prior_temporal_correlation.dtype == np.float32
    assert set(prior_spatial_correlation.dims) == {
        "subsector0",
        "sector0",
        "subsector1",
        "sector1",
    }
    assert prior_spatial_correlation.dtype == np.float32
    assert set(footprints.dims) == {"measurement", "state"}
    assert footprints.dtype == np.float32
    assert set(target_emissions.dims) == {"state"}
    assert target_emissions.dtype == np.float32
    assert set(measurements.dims) == {"measurement"}
    assert measurements.dtype == np.float32
    assert set(measurement_covariance.dims) == {"measurement0", "measurement1"}
    assert measurement_covariance.dtype == np.float32


def test_restack_coords():
    dataarray = xr.DataArray(
        np.random.randn(10, 5, 4, 2),
        dims=["measurement", "Time", "subsector", "sector"],
    ).stack(state=["Time", "subsector", "sector"])
    unstack_dims = ["state"]
    stack_dict = {"subsector_sector": ["subsector", "sector"]}
    dim_order = ["measurement", "Time", "subsector_sector"]

    restacked = _restack_coords(dataarray, unstack_dims, stack_dict, dim_order)

    assert isinstance(restacked, xr.DataArray)
    assert set(restacked.dims) == {"measurement", "Time", "subsector_sector"}
    assert restacked.shape == (10, 5, 4 * 2)
