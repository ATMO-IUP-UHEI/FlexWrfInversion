from pathlib import Path

import pytest
import xarray as xr

from flexwrfinversion.loaders.footprint import (
    FlexibleFootprintLoaderAnthBio,
    FlexibleFootprintLoaderAnthBioCo,
    FlexibleFootprintLoaderTotal,
    LoadFootprintAnthAndBioSectors,
    LoadFootprintAnthBioCO,
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


@pytest.fixture
def load_footprint_anth_bio_co():
    return LoadFootprintAnthBioCO(
        remapped_data_path=EXAMPLE_DIRECTORY_0 / "remapped_data",
        season="spring",
        city="munich",
        prior_type="true",
        time_resolution=3,
    )


@pytest.fixture
def flexible_footprint_loader_total():
    return FlexibleFootprintLoaderTotal(
        EXAMPLE_DIRECTORY_0
        / "remapped_data"
        / "spring"
        / "munich"
        / "true"
        / "remapped_footprints_sums_3H.nc",
        EXAMPLE_DIRECTORY_0
        / "remapped_data"
        / "spring"
        / "germany"
        / "true"
        / "remapped_footprints_sums_3H.nc",
    )


@pytest.fixture
def flexible_footprint_loader_total_keep_only():
    return FlexibleFootprintLoaderTotal(
        EXAMPLE_DIRECTORY_0
        / "remapped_data"
        / "spring"
        / "munich"
        / "true"
        / "remapped_footprints_sums_3H.nc",
        EXAMPLE_DIRECTORY_0
        / "remapped_data"
        / "spring"
        / "germany"
        / "true"
        / "remapped_footprints_sums_3H.nc",
        keep_only=["site00"],
    )


@pytest.fixture
def flexible_footprint_loader_total_leave_out():
    return FlexibleFootprintLoaderTotal(
        EXAMPLE_DIRECTORY_0
        / "remapped_data"
        / "spring"
        / "munich"
        / "true"
        / "remapped_footprints_sums_3H.nc",
        EXAMPLE_DIRECTORY_0
        / "remapped_data"
        / "spring"
        / "germany"
        / "true"
        / "remapped_footprints_sums_3H.nc",
        leave_out=["site01"],
    )


@pytest.fixture
def flexible_footprint_loader_total_times_of_day():
    return FlexibleFootprintLoaderTotal(
        EXAMPLE_DIRECTORY_0
        / "remapped_data"
        / "spring"
        / "munich"
        / "true"
        / "remapped_footprints_sums_3H.nc",
        EXAMPLE_DIRECTORY_0
        / "remapped_data"
        / "spring"
        / "germany"
        / "true"
        / "remapped_footprints_sums_3H.nc",
        times_of_day=[0, 4],
    )


@pytest.fixture
def flexible_footprint_loader_anth_bio():
    return FlexibleFootprintLoaderAnthBio(
        footprint_file_city_bio=EXAMPLE_DIRECTORY_0
        / "remapped_data"
        / "spring"
        / "munich"
        / "true"
        / "remapped_footprints_3H.nc",
        footprint_file_city_ant=EXAMPLE_DIRECTORY_0
        / "remapped_data"
        / "spring"
        / "munich"
        / "true"
        / "remapped_footprints_sums_3H.nc",
        footprint_file_germany_bio=EXAMPLE_DIRECTORY_0
        / "remapped_data"
        / "spring"
        / "germany"
        / "true"
        / "remapped_footprints_vprm_co_3H.nc",
        footprint_file_germany_ant=EXAMPLE_DIRECTORY_0
        / "remapped_data"
        / "spring"
        / "germany"
        / "true"
        / "remapped_footprints_sums_3H.nc",
    )


@pytest.fixture
def flexible_footprint_loader_anth_bio_co():
    return FlexibleFootprintLoaderAnthBioCo(
        footprint_file_city_bio=EXAMPLE_DIRECTORY_0
        / "remapped_data"
        / "spring"
        / "munich"
        / "true"
        / "remapped_footprints_3H.nc",
        footprint_file_city_ant=EXAMPLE_DIRECTORY_0
        / "remapped_data"
        / "spring"
        / "munich"
        / "true"
        / "remapped_footprints_sums_3H.nc",
        footprint_file_city_co=EXAMPLE_DIRECTORY_0
        / "remapped_data"
        / "spring"
        / "munich"
        / "true"
        / "remapped_footprints_3H.nc",
        footprint_file_germany_bio=EXAMPLE_DIRECTORY_0
        / "remapped_data"
        / "spring"
        / "germany"
        / "true"
        / "remapped_footprints_vprm_co_3H.nc",
        footprint_file_germany_ant=EXAMPLE_DIRECTORY_0
        / "remapped_data"
        / "spring"
        / "germany"
        / "true"
        / "remapped_footprints_sums_3H.nc",
        footprint_file_germany_co=EXAMPLE_DIRECTORY_0
        / "remapped_data"
        / "spring"
        / "germany"
        / "true"
        / "remapped_footprints_vprm_co_3H.nc",
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


class Test_LoadFootprintAnthBioCO:
    def test_footprint(self, load_footprint_anth_bio_co):
        footprint = load_footprint_anth_bio_co.footprint
        assert footprint is not None
        assert isinstance(footprint, xr.DataArray)
        assert len(footprint.dims) == 2
        assert set(footprint.dims) == {"state", "measurement"}
        assert {"sector", "Time", "MTime", "subsector", "MPlace", "species"}.issubset(
            set(footprint.coords.keys())
        )

    def test_load_timeframe(self, load_footprint_anth_bio_co):
        footprint = load_footprint_anth_bio_co.footprint
        start_time = footprint["Time"].values[0]
        end_time = footprint["Time"].values[3]
        start_mtime = footprint["MTime"].values[0]
        end_mtime = footprint["MTime"].values[4]

        footprint_timeframe = load_footprint_anth_bio_co.load_timeframe(
            start_time=start_time,
            end_time=end_time,
            start_mtime=start_mtime,
            end_mtime=end_mtime,
        )

        assert footprint_timeframe is not None
        assert isinstance(footprint_timeframe, xr.DataArray)
        assert len(footprint_timeframe.dims) == 2
        assert set(footprint_timeframe.dims) == {"state", "measurement"}
        assert {"sector", "Time", "MTime", "subsector", "MPlace", "species"}.issubset(
            set(footprint_timeframe.coords.keys())
        )


class Test_FlexibleFootprintLoaderTotal:
    def test_footprint(self, flexible_footprint_loader_total):
        footprint = flexible_footprint_loader_total.footprint
        assert footprint is not None
        assert isinstance(footprint, xr.DataArray)
        assert len(footprint.dims) == 2
        assert set(footprint.dims) == {"state", "measurement"}

    def test_footprint_keep_only_leave_out(
        self,
        flexible_footprint_loader_total_keep_only,
        flexible_footprint_loader_total_leave_out,
    ):
        for footprint in [
            flexible_footprint_loader_total_keep_only.footprint,
            flexible_footprint_loader_total_leave_out.footprint,
        ]:
            assert footprint is not None
            assert isinstance(footprint, xr.DataArray)
            assert len(footprint.dims) == 2
            assert set(footprint.dims) == {"state", "measurement"}
            assert set(footprint.MPlace.values) == {b"site00"}

    def test_footprint_times_of_day(self, flexible_footprint_loader_total_times_of_day):
        footprint = flexible_footprint_loader_total_times_of_day.footprint
        assert set(footprint.MTime.dt.hour.values) == {0, 4}

    def test_load_timeframe(self, flexible_footprint_loader_total):
        footprint = flexible_footprint_loader_total.footprint
        start_time = footprint["Time"].values[0]
        end_time = footprint["Time"].values[3]
        start_mtime = footprint["MTime"].values[0]
        end_mtime = footprint["MTime"].values[4]

        footprint_timeframe = flexible_footprint_loader_total.load_timeframe(
            start_time=start_time,
            end_time=end_time,
            start_mtime=start_mtime,
            end_mtime=end_mtime,
        )

        assert footprint_timeframe is not None
        assert isinstance(footprint_timeframe, xr.DataArray)
        assert len(footprint_timeframe.dims) == 2
        assert set(footprint_timeframe.dims) == {"state", "measurement"}


class Test_FlexibleFootprintLoaderAnthBio:
    def test_footprint(self, flexible_footprint_loader_anth_bio):
        footprint = flexible_footprint_loader_anth_bio.footprint
        assert footprint is not None
        assert isinstance(footprint, xr.DataArray)
        assert len(footprint.dims) == 2
        assert set(footprint.dims) == {"state", "measurement"}
        assert {"sector"}.issubset(set(footprint.coords.keys()))

    def test_load_timeframe(self, flexible_footprint_loader_anth_bio):
        footprint = flexible_footprint_loader_anth_bio.footprint
        start_time = footprint["Time"].values[0]
        end_time = footprint["Time"].values[3]
        start_mtime = footprint["MTime"].values[0]
        end_mtime = footprint["MTime"].values[4]

        footprint_timeframe = flexible_footprint_loader_anth_bio.load_timeframe(
            start_time=start_time,
            end_time=end_time,
            start_mtime=start_mtime,
            end_mtime=end_mtime,
        )

        assert footprint_timeframe is not None
        assert isinstance(footprint_timeframe, xr.DataArray)
        assert len(footprint_timeframe.dims) == 2
        assert set(footprint_timeframe.dims) == {"state", "measurement"}
        assert {"sector"}.issubset(set(footprint_timeframe.coords.keys()))


class Test_FlexibleFootprintLoaderAnthBioCo:
    def test_footprint(self, flexible_footprint_loader_anth_bio_co):
        footprint = flexible_footprint_loader_anth_bio_co.footprint
        assert footprint is not None
        assert isinstance(footprint, xr.DataArray)
        assert len(footprint.dims) == 2
        assert set(footprint.dims) == {"state", "measurement"}
        assert {"sector", "Time", "MTime", "subsector", "MPlace", "species"}.issubset(
            set(footprint.coords.keys())
        )

    def test_load_timeframe(self, flexible_footprint_loader_anth_bio_co):
        footprint = flexible_footprint_loader_anth_bio_co.footprint
        start_time = footprint["Time"].values[0]
        end_time = footprint["Time"].values[3]
        start_mtime = footprint["MTime"].values[0]
        end_mtime = footprint["MTime"].values[4]

        footprint_timeframe = flexible_footprint_loader_anth_bio_co.load_timeframe(
            start_time=start_time,
            end_time=end_time,
            start_mtime=start_mtime,
            end_mtime=end_mtime,
        )

        assert footprint_timeframe is not None
        assert isinstance(footprint_timeframe, xr.DataArray)
        assert len(footprint_timeframe.dims) == 2
        assert set(footprint_timeframe.dims) == {"state", "measurement"}
        assert {"sector", "Time", "MTime", "subsector", "MPlace", "species"}.issubset(
            set(footprint_timeframe.coords.keys())
        )
