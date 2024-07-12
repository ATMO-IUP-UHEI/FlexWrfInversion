from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from flexwrfinversion.loaders.prior import FlatPrior, ShiftToBiospheric
from flexwrfinversion.loaders.prior_covariance import (
    RelativeErrorWithSpatialCorrelation,
    TargetAsErrorNoCorrelation,
    TargetAsErrorWithCO_Correlation,
)
from flexwrfinversion.loaders.target import (
    TargetLoaderAnthAndBioSectors,
    TargetLoaderAnthBioCO,
    TargetLoaderTotalInCity,
)

EXAMPLE_DIRECTORY_0 = Path(__file__).parent.parent / "data" / "example_directory_0"


@pytest.fixture
def shift_to_biospheric():
    target_loader = TargetLoaderTotalInCity(
        remapped_data_path=EXAMPLE_DIRECTORY_0 / "remapped_data",
        season="spring",
        city="munich",
        prior_type="true",
        time_resolution=3,
    )
    return ShiftToBiospheric(target_loader=target_loader)


# fixture for RelativeErrorWithSpatialCorrelation
@pytest.fixture
def relative_error_with_spatial_correlation():
    target_loader = TargetLoaderTotalInCity(
        remapped_data_path=EXAMPLE_DIRECTORY_0 / "remapped_data",
        season="spring",
        city="munich",
        prior_type="true",
        time_resolution=3,
    )
    shift_to_biospheric = ShiftToBiospheric(target_loader=target_loader)
    return RelativeErrorWithSpatialCorrelation(
        prior_loader=shift_to_biospheric,
        relative_error=0.5,
    )


@pytest.fixture
def target_as_error_no_correlation():
    target_loader = TargetLoaderAnthAndBioSectors(
        remapped_data_path=EXAMPLE_DIRECTORY_0 / "remapped_data",
        season="spring",
        city="munich",
        prior_type="true",
        time_resolution=3,
    )
    flat_prior = FlatPrior(target_loader=target_loader, value=0.1)
    return TargetAsErrorNoCorrelation(
        prior_loader=flat_prior,
    )


@pytest.fixture
def target_as_error_with_co_correlation():
    target_loader = TargetLoaderAnthBioCO(
        remapped_data_path=EXAMPLE_DIRECTORY_0 / "remapped_data",
        season="spring",
        city="munich",
        prior_type="true",
        time_resolution=3,
    )
    flat_prior = FlatPrior(target_loader=target_loader, value=0)
    return TargetAsErrorWithCO_Correlation(
        prior_loader=flat_prior,
        anth_co_correlation=0.5,
    )


class Test_RelativeErrorWithSpatialCorrelation:
    @pytest.mark.parametrize("relative_error", [0.5, 1])
    def test_prior_std(self, shift_to_biospheric, relative_error):
        prior_covariance_loader = RelativeErrorWithSpatialCorrelation(
            prior_loader=shift_to_biospheric,
            relative_error=relative_error,
        )
        assert prior_covariance_loader.prior_std is not None
        assert isinstance(prior_covariance_loader.prior_std, xr.DataArray)
        assert len(prior_covariance_loader.prior_std.dims) == 1
        assert (prior_covariance_loader.prior_std >= 0).all()
        assert np.allclose(
            prior_covariance_loader.prior_std,
            np.abs(shift_to_biospheric.prior) * relative_error,
        )

    def test_load_timeframe(self, relative_error_with_spatial_correlation):
        start_time = relative_error_with_spatial_correlation.prior_std.Time.values[0]
        end_time = relative_error_with_spatial_correlation.prior_std.Time.values[4]
        prior_covariance = relative_error_with_spatial_correlation.load_timeframe(
            start_time, end_time
        )
        assert prior_covariance is not None
        assert isinstance(prior_covariance, xr.DataArray)
        assert len(prior_covariance.dims) == 2
        assert ((prior_covariance != 0) == np.eye(prior_covariance.shape[0])).all()
        assert set(prior_covariance.dims) == {"state0", "state1"}

    def test_spatial_correlation(self, relative_error_with_spatial_correlation):
        assert relative_error_with_spatial_correlation.spatial_correlation is not None
        assert isinstance(
            relative_error_with_spatial_correlation.spatial_correlation, xr.DataArray
        )
        assert (
            len(relative_error_with_spatial_correlation.spatial_correlation.dims) == 2
        )
        assert (
            relative_error_with_spatial_correlation.spatial_correlation
            == np.eye(
                relative_error_with_spatial_correlation.spatial_correlation.shape[0]
            )
        ).all()
        assert set(
            relative_error_with_spatial_correlation.spatial_correlation.dims
        ) == {
            "subsector0",
            "subsector1",
        }


class Test_TargetAsErrorNoCorrelation:
    def test_prior_std(self, target_as_error_no_correlation):
        prior_std = target_as_error_no_correlation.prior_std
        assert prior_std is not None
        assert isinstance(prior_std, xr.DataArray)
        assert len(prior_std.dims) == 1
        assert (prior_std >= 0).all()
        assert np.allclose(
            prior_std,
            np.abs(target_as_error_no_correlation.prior_loader.target_loader.target),
        )

    def test_load_timeframe(self, target_as_error_no_correlation):
        start_time = target_as_error_no_correlation.prior_std.Time.values[0]
        end_time = target_as_error_no_correlation.prior_std.Time.values[4]
        prior_covariance = target_as_error_no_correlation.load_timeframe(
            start_time, end_time
        )
        assert prior_covariance is not None
        assert isinstance(prior_covariance, xr.DataArray)
        assert len(prior_covariance.dims) == 2
        assert ((prior_covariance != 0) == np.eye(prior_covariance.shape[0])).all()
        assert set(prior_covariance.dims) == {"state0", "state1"}


class Test_TargetAsErrorWithCO_Correlation:
    def test_prior_std(self, target_as_error_with_co_correlation):
        prior_std = target_as_error_with_co_correlation.prior_std
        assert prior_std is not None
        assert isinstance(prior_std, xr.DataArray)
        assert len(prior_std.dims) == 1
        assert (prior_std >= 0).all()
        assert np.allclose(
            prior_std,
            np.abs(
                target_as_error_with_co_correlation.prior_loader.target_loader.target
            ),
        )

    def test_spatial_correlation(self, target_as_error_with_co_correlation):
        spatial_correlation = target_as_error_with_co_correlation.spatial_correlation
        assert spatial_correlation is not None
        assert isinstance(spatial_correlation, xr.DataArray)
        assert len(spatial_correlation.dims) == 2
        assert (spatial_correlation == np.eye(spatial_correlation.shape[0])).all()
        assert set(spatial_correlation.dims) == {
            "subsector0",
            "subsector1",
        }

    def test_sector_correlation(self, target_as_error_with_co_correlation):
        sector_correlation = target_as_error_with_co_correlation.sector_correlation
        assert sector_correlation is not None
        assert isinstance(sector_correlation, xr.DataArray)
        assert len(sector_correlation.dims) == 2
        assert (np.diag(sector_correlation) == 1).all()
        assert set(sector_correlation.dims) == {
            "sector0",
            "sector1",
        }
        assert (
            sector_correlation.unstack().sel(sector0="CO2_ANT_TOTAL", sector1="E_CO")
            == 0.5
        )
        assert (
            sector_correlation.unstack().sel(sector0="E_CO", sector1="CO2_ANT_TOTAL")
            == 0.5
        )
        assert np.count_nonzero(sector_correlation == 0.5) == 2

    def test_load_timeframe(self, target_as_error_with_co_correlation):
        start_time = target_as_error_with_co_correlation.prior_std.Time.values[0]
        end_time = target_as_error_with_co_correlation.prior_std.Time.values[4]
        prior_covariance = target_as_error_with_co_correlation.load_timeframe(
            start_time, end_time
        )
        assert prior_covariance is not None
        assert isinstance(prior_covariance, xr.DataArray)
        assert len(prior_covariance.dims) == 2
        assert set(prior_covariance.dims) == {"state0", "state1"}
