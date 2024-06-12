from pathlib import Path

import pytest
import xarray as xr

from flexwrfinversion.loaders.target import TargetLoaderTotalInCity

EXAMPLE_DIRECTORY_0 = Path(__file__).parent.parent / "data" / "example_directory_0"


@pytest.fixture
def target_loader_total_in_city():
    return TargetLoaderTotalInCity(
        remapped_data_path=EXAMPLE_DIRECTORY_0 / "remapped_data",
        season="spring",
        city="munich",
        prior_type="true",
        time_resolution=3,
    )


class Test_TargetLoaderTotalInCity:
    def test_target_with_additional_sectors(self, target_loader_total_in_city):
        assert target_loader_total_in_city.target_with_additional_sectors is not None
        assert isinstance(
            target_loader_total_in_city.target_with_additional_sectors, xr.Dataset
        )
        assert set(
            target_loader_total_in_city.target_with_additional_sectors.data_vars
        ) == {"CO2_ANT_TOTAL", "E_CO2_VPRM", "CO2_TOTAL"}
        assert len(target_loader_total_in_city.target_with_additional_sectors.dims) == 1
        assert set(target_loader_total_in_city.target_with_additional_sectors.dims) == {
            "state"
        }

    def test_target(self, target_loader_total_in_city):
        assert target_loader_total_in_city.target is not None
        assert isinstance(target_loader_total_in_city.target, xr.DataArray)
        assert len(target_loader_total_in_city.target.dims) == 1
        assert set(target_loader_total_in_city.target.dims) == {"state"}

    def test_load_timeframe(self, target_loader_total_in_city):
        target = target_loader_total_in_city.target
        start_time = target["Time"].values[0]
        end_time = target["Time"].values[3]

        target_timeframe = target_loader_total_in_city.load_timeframe(
            start_time=start_time,
            end_time=end_time,
        )

        assert target_timeframe is not None
        assert isinstance(target_timeframe, xr.DataArray)
        assert len(target_timeframe.dims) == 1
        assert set(target_timeframe.dims) == {"state"}
