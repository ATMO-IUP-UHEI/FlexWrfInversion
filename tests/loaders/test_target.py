from pathlib import Path

import pytest
import xarray as xr

from flexwrfinversion.loaders.target import (
    FlexibleTargetLoaderAnthBio,
    FlexibleTargetLoaderAnthBioCo,
    FlexibleTargetLoaderTotal,
    TargetLoaderAnthAndBioSectors,
    TargetLoaderAnthBioCO,
    TargetLoaderTotalInCity,
)

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


@pytest.fixture
def target_loader_anth_and_bio_sectors():
    return TargetLoaderAnthAndBioSectors(
        remapped_data_path=EXAMPLE_DIRECTORY_0 / "remapped_data",
        season="spring",
        city="munich",
        prior_type="true",
        time_resolution=3,
    )


@pytest.fixture
def target_loader_anth_bio_co():
    return TargetLoaderAnthBioCO(
        remapped_data_path=EXAMPLE_DIRECTORY_0 / "remapped_data",
        season="spring",
        city="munich",
        prior_type="true",
        time_resolution=3,
    )


@pytest.fixture
def flexible_target_loader_total():
    return FlexibleTargetLoaderTotal(
        EXAMPLE_DIRECTORY_0
        / "remapped_data"
        / "spring"
        / "munich"
        / "true"
        / "remapped_true_emissions_sums_3H.nc",
        EXAMPLE_DIRECTORY_0
        / "remapped_data"
        / "spring"
        / "germany"
        / "true"
        / "remapped_true_emissions_sums_3H.nc",
    )


@pytest.fixture
def flexible_target_loader_anth_bio():
    return FlexibleTargetLoaderAnthBio(
        target_file_city_bio=EXAMPLE_DIRECTORY_0
        / "remapped_data"
        / "spring"
        / "munich"
        / "true"
        / "remapped_true_emissions_3H.nc",
        target_file_city_ant=EXAMPLE_DIRECTORY_0
        / "remapped_data"
        / "spring"
        / "munich"
        / "true"
        / "remapped_true_emissions_sums_3H.nc",
        target_file_germany_bio=EXAMPLE_DIRECTORY_0
        / "remapped_data"
        / "spring"
        / "germany"
        / "true"
        / "remapped_true_emissions_vprm_co_3H.nc",
        target_file_germany_ant=EXAMPLE_DIRECTORY_0
        / "remapped_data"
        / "spring"
        / "germany"
        / "true"
        / "remapped_true_emissions_sums_3H.nc",
    )


@pytest.fixture
def flexible_target_loader_anth_bio_co():
    return FlexibleTargetLoaderAnthBioCo(
        target_file_city_bio=EXAMPLE_DIRECTORY_0
        / "remapped_data"
        / "spring"
        / "munich"
        / "true"
        / "remapped_true_emissions_3H.nc",
        target_file_city_ant=EXAMPLE_DIRECTORY_0
        / "remapped_data"
        / "spring"
        / "munich"
        / "true"
        / "remapped_true_emissions_sums_3H.nc",
        target_file_city_co=EXAMPLE_DIRECTORY_0
        / "remapped_data"
        / "spring"
        / "munich"
        / "true"
        / "remapped_true_emissions_3H.nc",
        target_file_germany_bio=EXAMPLE_DIRECTORY_0
        / "remapped_data"
        / "spring"
        / "germany"
        / "true"
        / "remapped_true_emissions_vprm_co_3H.nc",
        target_file_germany_ant=EXAMPLE_DIRECTORY_0
        / "remapped_data"
        / "spring"
        / "germany"
        / "true"
        / "remapped_true_emissions_sums_3H.nc",
        target_file_germany_co=EXAMPLE_DIRECTORY_0
        / "remapped_data"
        / "spring"
        / "germany"
        / "true"
        / "remapped_true_emissions_vprm_co_3H.nc",
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


class Test_TargetLoaderAnthAndBioSectors:
    def test_target(self, target_loader_anth_and_bio_sectors):
        target = target_loader_anth_and_bio_sectors.target
        assert target is not None
        assert isinstance(target, xr.DataArray)
        assert len(target.dims) == 1
        assert set(target.dims) == {"state"}
        assert {"subsector", "Time", "sector"}.issubset(set(target.coords.keys()))

    def test_load_timeframe(self, target_loader_anth_and_bio_sectors):
        target = target_loader_anth_and_bio_sectors.target
        start_time = target["Time"].values[0]
        end_time = target["Time"].values[3]

        target_timeframe = target_loader_anth_and_bio_sectors.load_timeframe(
            start_time=start_time,
            end_time=end_time,
        )

        assert target_timeframe is not None
        assert isinstance(target_timeframe, xr.DataArray)
        assert len(target_timeframe.dims) == 1
        assert set(target_timeframe.dims) == {"state"}


class Test_TargetLoaderAnthBioCO:
    def test_target(self, target_loader_anth_bio_co):
        target = target_loader_anth_bio_co.target
        assert target is not None
        assert isinstance(target, xr.DataArray)
        assert len(target.dims) == 1
        assert set(target.dims) == {"state"}
        assert {"subsector", "Time", "sector"}.issubset(set(target.coords.keys()))
        assert {"CO2_ANT_TOTAL", "E_CO2_VPRM", "E_CO"} == set(target.sector.values)

    def test_load_timeframe(self, target_loader_anth_bio_co):
        target = target_loader_anth_bio_co.target
        start_time = target["Time"].values[0]
        end_time = target["Time"].values[3]

        target_timeframe = target_loader_anth_bio_co.load_timeframe(
            start_time=start_time,
            end_time=end_time,
        )

        assert target_timeframe is not None
        assert isinstance(target_timeframe, xr.DataArray)
        assert len(target_timeframe.dims) == 1
        assert set(target_timeframe.dims) == {"state"}
        assert {"subsector", "Time", "sector"}.issubset(set(target.coords.keys()))
        assert {"CO2_ANT_TOTAL", "E_CO2_VPRM", "E_CO"} == set(target.sector.values)


class Test_FlexibleTargetLoaderTotal:
    def test_target(self, flexible_target_loader_total):
        target = flexible_target_loader_total.target
        assert target is not None
        assert isinstance(target, xr.DataArray)
        assert len(target.dims) == 1
        assert set(target.dims) == {"state"}

    def test_load_timeframe(self, flexible_target_loader_total):
        target = flexible_target_loader_total.target
        start_time = target["Time"].values[0]
        end_time = target["Time"].values[3]

        target_timeframe = flexible_target_loader_total.load_timeframe(
            start_time=start_time,
            end_time=end_time,
        )

        assert target_timeframe is not None
        assert isinstance(target_timeframe, xr.DataArray)
        assert len(target_timeframe.dims) == 1
        assert set(target_timeframe.dims) == {"state"}


class Test_FlexibleTargetLoaderAnthBio:
    def test_target(self, flexible_target_loader_anth_bio):
        target = flexible_target_loader_anth_bio.target
        assert target is not None
        assert isinstance(target, xr.DataArray)
        assert len(target.dims) == 1
        assert set(target.dims) == {"state"}

    def test_load_timeframe(self, flexible_target_loader_anth_bio):
        target = flexible_target_loader_anth_bio.target
        start_time = target["Time"].values[0]
        end_time = target["Time"].values[3]

        target_timeframe = flexible_target_loader_anth_bio.load_timeframe(
            start_time=start_time,
            end_time=end_time,
        )

        assert target_timeframe is not None
        assert isinstance(target_timeframe, xr.DataArray)
        assert len(target_timeframe.dims) == 1
        assert set(target_timeframe.dims) == {"state"}


class Test_FlexibleTargetLoaderAnthBioCo:
    def test_target(self, flexible_target_loader_anth_bio_co):
        target = flexible_target_loader_anth_bio_co.target
        assert target is not None
        assert isinstance(target, xr.DataArray)
        assert len(target.dims) == 1
        assert set(target.dims) == {"state"}
        assert set(target.unstack().dims) == {"subsector", "sector", "Time"}

    def test_load_timeframe(self, flexible_target_loader_anth_bio_co):
        target = flexible_target_loader_anth_bio_co.target
        start_time = target["Time"].values[0]
        end_time = target["Time"].values[3]

        target_timeframe = flexible_target_loader_anth_bio_co.load_timeframe(
            start_time=start_time,
            end_time=end_time,
        )

        assert target_timeframe is not None
        assert isinstance(target_timeframe, xr.DataArray)
        assert len(target_timeframe.dims) == 1
        assert set(target_timeframe.dims) == {"state"}
