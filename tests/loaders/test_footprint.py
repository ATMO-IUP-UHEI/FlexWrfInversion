from pathlib import Path

import pytest
import xarray as xr

from flexwrfinversion.loaders.footprint import (
    LoadFootprintAnthAndBioSectors,
    LoadFootprintForTotalInCity,
)

EXAMPLE_DIRECTORY_0 = Path(__file__).parent.parent / "data" / "example_directory_0"


@pytest.fixture
def load_footprint_for_total_in_city():
    return LoadFootprintForTotalInCity(
        remapped_data_path=EXAMPLE_DIRECTORY_0 / "remapped_data",
        season="spring",
        city="munich",
        prior_type="true",
        time_resolution=3,
    )


@pytest.fixture
def load_footprint_anth_and_bio_sectors():
    return LoadFootprintAnthAndBioSectors(
        remapped_data_path=EXAMPLE_DIRECTORY_0 / "remapped_data",
        season="spring",
        city="munich",
        prior_type="true",
        time_resolution=3,
    )


class Test_LoadFootprintForTotalInCity:
    def test_footprint(self, load_footprint_for_total_in_city):
        footprint = load_footprint_for_total_in_city.footprint
        assert footprint is not None
        assert isinstance(footprint, xr.DataArray)
        assert len(footprint.dims) == 2
        assert set(footprint.dims) == {"state", "measurement"}

    def test_load_timeframe(self, load_footprint_for_total_in_city):
        footprint = load_footprint_for_total_in_city.footprint
        start_time = footprint["Time"].values[0]
        end_time = footprint["Time"].values[3]
        start_mtime = footprint["MTime"].values[0]
        end_mtime = footprint["MTime"].values[4]

        footprint_timeframe = load_footprint_for_total_in_city.load_timeframe(
            start_time=start_time,
            end_time=end_time,
            start_mtime=start_mtime,
            end_mtime=end_mtime,
        )

        assert footprint_timeframe is not None
        assert isinstance(footprint_timeframe, xr.DataArray)
        assert len(footprint_timeframe.dims) == 2
        assert set(footprint_timeframe.dims) == {"state", "measurement"}


class Test_LoadFootprintAnthAndBioSectors:
    def test_footprint(self, load_footprint_anth_and_bio_sectors):
        footprint = load_footprint_anth_and_bio_sectors.footprint
        assert footprint is not None
        assert isinstance(footprint, xr.DataArray)
        assert len(footprint.dims) == 2
        assert set(footprint.dims) == {"state", "measurement"}
        assert {"sector"}.issubset(set(footprint.coords.keys()))

    def test_load_timeframe(self, load_footprint_anth_and_bio_sectors):
        footprint = load_footprint_anth_and_bio_sectors.footprint
        start_time = footprint["Time"].values[0]
        end_time = footprint["Time"].values[3]
        start_mtime = footprint["MTime"].values[0]
        end_mtime = footprint["MTime"].values[4]

        footprint_timeframe = load_footprint_anth_and_bio_sectors.load_timeframe(
            start_time=start_time,
            end_time=end_time,
            start_mtime=start_mtime,
            end_mtime=end_mtime,
        )

        assert footprint_timeframe is not None
        assert isinstance(footprint_timeframe, xr.DataArray)
        assert len(footprint_timeframe.dims) == 2
        assert set(footprint_timeframe.dims) == {"state", "measurement"}
        assert {"sector"}.issubset(set(footprint.coords.keys()))
