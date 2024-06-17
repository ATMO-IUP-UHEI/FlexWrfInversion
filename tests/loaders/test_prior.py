from pathlib import Path

import pytest
import xarray as xr

from flexwrfinversion.loaders.prior import FlatPrior, ShiftToBiospheric
from flexwrfinversion.loaders.target import (
    TargetLoaderAnthAndBioSectors,
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


@pytest.fixture
def flat_prior():
    target_loader = TargetLoaderAnthAndBioSectors(
        remapped_data_path=EXAMPLE_DIRECTORY_0 / "remapped_data",
        season="spring",
        city="munich",
        prior_type="true",
        time_resolution=3,
    )
    return FlatPrior(target_loader=target_loader, value=0.1)


class Test_ShiftToBiospheric:
    def test_prior(self, shift_to_biospheric):
        assert shift_to_biospheric.prior is not None
        assert isinstance(shift_to_biospheric.prior, xr.DataArray)
        assert len(shift_to_biospheric.prior.dims) == 1
        assert (
            shift_to_biospheric.prior != shift_to_biospheric.target_loader.target
        ).any()
        assert (
            (shift_to_biospheric.prior > 0)
            == (shift_to_biospheric.target_loader.target > 0)
        ).all()
        assert set(shift_to_biospheric.prior.dims) == {"state"}

    def test_load_timeframe(self, shift_to_biospheric):
        prior = shift_to_biospheric.prior
        start_time = prior["Time"].values[0]
        end_time = prior["Time"].values[3]

        prior_timeframe = shift_to_biospheric.load_timeframe(
            start_time=start_time,
            end_time=end_time,
        )

        assert prior_timeframe is not None
        assert isinstance(prior_timeframe, xr.DataArray)
        assert len(prior_timeframe.dims) == 1
        assert set(prior_timeframe.dims) == {"state"}


class Test_FlatPrior:
    def test_prior(self, flat_prior):
        assert flat_prior.prior is not None
        assert isinstance(flat_prior.prior, xr.DataArray)
        assert len(flat_prior.prior.dims) == 1
        assert set(flat_prior.prior.dims) == {"state"}
        assert flat_prior.prior.max().item() == pytest.approx(0.1, abs=1e-6)

    def test_load_timeframe(self, flat_prior):
        prior = flat_prior.prior
        start_time = prior["Time"].values[0]
        end_time = prior["Time"].values[3]

        prior_timeframe = flat_prior.load_timeframe(
            start_time=start_time,
            end_time=end_time,
        )

        assert prior_timeframe is not None
        assert isinstance(prior_timeframe, xr.DataArray)
        assert len(prior_timeframe.dims) == 1
        assert set(prior_timeframe.dims) == {"state"}
        assert (prior_timeframe == 0.1).all()
