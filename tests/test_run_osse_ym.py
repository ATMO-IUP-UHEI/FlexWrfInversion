import numpy as np
import pandas as pd
import xarray as xr

from flexwrfinversion.run_osse_ym import (
    _initialize_loaders,
    _load_inversion_data,
    _restack_coords,
    _run_inversion_ym,
    _setup_restacking,
)


def test_run_inversion_ym():
    # Create synthetic data
    mplace = np.char.encode(np.array(["site1", "site2", "site3"]))
    mtime = pd.date_range("2020-01-01", periods=10, freq="h").to_numpy()
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

    result = _run_inversion_ym(
        sites,
        prior_emissions.astype(np.float32),
        prior_standard_deviation.astype(np.float32),
        prior_temporal_correlation.astype(np.float32),
        prior_spatial_correlation.astype(np.float32),
        footprints.astype(np.float32),
        measurements.astype(np.float32),
        measurement_covariance.astype(np.float32),
        target_emissions.astype(np.float32),
    )

    assert isinstance(result, xr.Dataset)
    assert "posterior_emissions" in result
    assert "posterior_std" in result

    posterior_emissions = result.posterior_emissions
    posterior_std = result.posterior_std

    assert (
        set(posterior_emissions.dims)
        == set(prior_emissions.dims)
        == {"Time", "subsector"}
    )
    assert (
        set(posterior_std.dims)
        == set(prior_standard_deviation.dims)
        == {"Time", "subsector"}
    )
    for dim in result.posterior_emissions.dims:
        assert (result.posterior_emissions[dim] == prior_emissions[dim]).all()
        assert (result.posterior_std[dim] == prior_standard_deviation[dim]).all()


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
        )

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
    assert set(prior_spatial_correlation.dims) == {"subsector0", "subsector1"}
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
