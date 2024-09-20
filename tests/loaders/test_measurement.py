from pathlib import Path

import pytest
import xarray as xr

from flexwrfinversion.loaders.footprint import (
    FlexibleFootprintLoaderAnthBioCo,
    FlexibleFootprintLoaderTotal,
    LoadFootprintAnthBioCO,
    LoadFootprintForTotalInCity,
)
from flexwrfinversion.loaders.measurement import (
    FlexibleMeasurementLoaderTotal,
    FlexibleMeasurementLoaderTotalCo,
    MeasurementFromFile,
    MeasurementFromFileCO,
)
from flexwrfinversion.loaders.target import (
    FlexibleTargetLoaderTotal,
    TargetLoaderAnthBioCO,
    TargetLoaderTotalInCity,
)

EXAMPLE_DIRECTORY_0 = Path(__file__).parent.parent / "data" / "example_directory_0"


@pytest.fixture
def measurement_from_file():
    return MeasurementFromFile(
        target_loader=TargetLoaderTotalInCity(
            remapped_data_path=EXAMPLE_DIRECTORY_0 / "remapped_data",
            season="spring",
            city="munich",
            prior_type="true",
            time_resolution=3,
        ),
        footprint_loader=LoadFootprintForTotalInCity(
            remapped_data_path=EXAMPLE_DIRECTORY_0 / "remapped_data",
            season="spring",
            city="munich",
            prior_type="true",
            time_resolution=3,
        ),
    )


@pytest.fixture
def measurement_from_file_co():
    return MeasurementFromFileCO(
        target_loader=TargetLoaderAnthBioCO(
            remapped_data_path=EXAMPLE_DIRECTORY_0 / "remapped_data",
            season="spring",
            city="munich",
            prior_type="true",
            time_resolution=3,
        ),
        footprint_loader=LoadFootprintAnthBioCO(
            remapped_data_path=EXAMPLE_DIRECTORY_0 / "remapped_data",
            season="spring",
            city="munich",
            prior_type="true",
            time_resolution=3,
        ),
    )


@pytest.fixture
def flexible_measurement_loader_total():
    return FlexibleMeasurementLoaderTotal(
        target_loader=FlexibleTargetLoaderTotal(
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
        ),
        footprint_loader=FlexibleFootprintLoaderTotal(
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
        ),
        measurement_file_city=EXAMPLE_DIRECTORY_0
        / "remapped_data"
        / "spring"
        / "munich"
        / "true"
        / "true_concentrations_sums.nc",
        measurement_file_germany=EXAMPLE_DIRECTORY_0
        / "remapped_data"
        / "spring"
        / "germany"
        / "true"
        / "true_concentrations_sums.nc",
    )


@pytest.fixture
def flexible_measurement_loader_total_keep_only():
    return FlexibleMeasurementLoaderTotal(
        target_loader=FlexibleTargetLoaderTotal(
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
        ),
        footprint_loader=FlexibleFootprintLoaderTotal(
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
        ),
        measurement_file_city=EXAMPLE_DIRECTORY_0
        / "remapped_data"
        / "spring"
        / "munich"
        / "true"
        / "true_concentrations_sums.nc",
        measurement_file_germany=EXAMPLE_DIRECTORY_0
        / "remapped_data"
        / "spring"
        / "germany"
        / "true"
        / "true_concentrations_sums.nc",
        keep_only=["site00"],
    )


@pytest.fixture
def flexible_measurement_loader_total_leave_out():
    return FlexibleMeasurementLoaderTotal(
        target_loader=FlexibleTargetLoaderTotal(
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
        ),
        footprint_loader=FlexibleFootprintLoaderTotal(
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
        ),
        measurement_file_city=EXAMPLE_DIRECTORY_0
        / "remapped_data"
        / "spring"
        / "munich"
        / "true"
        / "true_concentrations_sums.nc",
        measurement_file_germany=EXAMPLE_DIRECTORY_0
        / "remapped_data"
        / "spring"
        / "germany"
        / "true"
        / "true_concentrations_sums.nc",
        leave_out=["site01"],
    )


@pytest.fixture
def flexible_measurement_loader_total_co():
    return FlexibleMeasurementLoaderTotalCo(
        target_loader=FlexibleTargetLoaderTotal(
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
        ),
        footprint_loader=FlexibleFootprintLoaderAnthBioCo(
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
        ),
        measurement_file_city_co2=EXAMPLE_DIRECTORY_0
        / "remapped_data"
        / "spring"
        / "munich"
        / "true"
        / "true_concentrations_sums.nc",
        measurement_file_city_co=EXAMPLE_DIRECTORY_0
        / "remapped_data"
        / "spring"
        / "munich"
        / "true"
        / "true_concentrations.nc",
        measurement_file_germany_co2=EXAMPLE_DIRECTORY_0
        / "remapped_data"
        / "spring"
        / "germany"
        / "true"
        / "true_concentrations_sums.nc",
        measurement_file_germany_co=EXAMPLE_DIRECTORY_0
        / "remapped_data"
        / "spring"
        / "germany"
        / "true"
        / "true_concentrations_vprm_co.nc",
    )


class Test_MeasurementFromFile:
    def test_measurements(self, measurement_from_file):
        measurements = measurement_from_file.measurements
        assert measurements is not None
        assert isinstance(measurements, xr.DataArray)
        assert len(measurements.dims) == 1
        assert set(measurements.dims) == {"measurement"}

    def test_load_timeframe(self, measurement_from_file):
        measurements = measurement_from_file.measurements
        start_mtime = measurements["MTime"].values[0]
        end_mtime = measurements["MTime"].values[4]

        measurements_timeframe = measurement_from_file.load_timeframe(
            start_time=start_mtime,
            end_time=end_mtime,
        )

        assert measurements_timeframe is not None
        assert isinstance(measurements_timeframe, xr.DataArray)
        assert len(measurements_timeframe.dims) == 1
        assert set(measurements_timeframe.dims) == {"measurement"}


class Test_MeasurementFromFileCO:
    def test_measurements(self, measurement_from_file_co):
        measurements = measurement_from_file_co.measurements
        assert measurements is not None
        assert isinstance(measurements, xr.DataArray)
        assert len(measurements.dims) == 1
        assert set(measurements.dims) == {"measurement"}
        assert {"MTime", "MPlace", "species"}.issubset(set(measurements.coords.keys()))
        assert set(measurements.unstack().species.values) == {
            "CO",
            "CO2",
        }

    def test_load_timeframe(self, measurement_from_file_co):
        measurements = measurement_from_file_co.measurements
        start_mtime = measurements["MTime"].values[0]
        end_mtime = measurements["MTime"].values[4]

        measurements_timeframe = measurement_from_file_co.load_timeframe(
            start_time=start_mtime,
            end_time=end_mtime,
        )

        assert measurements_timeframe is not None
        assert isinstance(measurements_timeframe, xr.DataArray)
        assert len(measurements_timeframe.dims) == 1
        assert set(measurements_timeframe.dims) == {"measurement"}
        assert {"MTime", "MPlace", "species"}.issubset(
            set(measurements_timeframe.coords.keys())
        )
        assert set(measurements_timeframe.unstack().species.values) == {
            "CO",
            "CO2",
        }


class Test_FlexibleMeasurementLoaderTotal:
    def test_measurements(self, flexible_measurement_loader_total):
        measurements = flexible_measurement_loader_total.measurements
        assert measurements is not None
        assert isinstance(measurements, xr.DataArray)
        assert len(measurements.dims) == 1
        assert set(measurements.dims) == {"measurement"}

    def test_measurements_keep_only_leave_out(
        self,
        flexible_measurement_loader_total_keep_only,
        flexible_measurement_loader_total_leave_out,
    ):
        for measurements in [
            flexible_measurement_loader_total_keep_only.measurements,
            flexible_measurement_loader_total_leave_out.measurements,
        ]:
            assert measurements is not None
            assert isinstance(measurements, xr.DataArray)
            assert len(measurements.dims) == 1
            assert set(measurements.dims) == {"measurement"}
            assert set(measurements.MPlace.values) == {b"site00"}

    def test_load_timeframe(self, flexible_measurement_loader_total):
        measurements = flexible_measurement_loader_total.measurements
        start_mtime = measurements["MTime"].values[0]
        end_mtime = measurements["MTime"].values[4]

        measurements_timeframe = flexible_measurement_loader_total.load_timeframe(
            start_time=start_mtime,
            end_time=end_mtime,
        )

        assert measurements_timeframe is not None
        assert isinstance(measurements_timeframe, xr.DataArray)
        assert len(measurements_timeframe.dims) == 1
        assert set(measurements_timeframe.dims) == {"measurement"}


class Test_FlexibleMeasurementLoaderTotalCo:
    def test_measurements(self, flexible_measurement_loader_total_co):
        measurements = flexible_measurement_loader_total_co.measurements
        assert measurements is not None
        assert isinstance(measurements, xr.DataArray)
        assert len(measurements.dims) == 1
        assert set(measurements.dims) == {"measurement"}
        assert set(measurements.unstack().dims) == {"species", "MPlace", "MTime"}

    def test_load_timeframe(self, flexible_measurement_loader_total_co):
        measurements = flexible_measurement_loader_total_co.measurements
        start_mtime = measurements["MTime"].values[0]
        end_mtime = measurements["MTime"].values[4]

        measurements_timeframe = flexible_measurement_loader_total_co.load_timeframe(
            start_time=start_mtime,
            end_time=end_mtime,
        )

        assert measurements_timeframe is not None
        assert isinstance(measurements_timeframe, xr.DataArray)
        assert len(measurements_timeframe.dims) == 1
        assert set(measurements_timeframe.dims) == {"measurement"}
        assert set(measurements_timeframe.unstack().dims) == {
            "species",
            "MPlace",
            "MTime",
        }
