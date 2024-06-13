from pathlib import Path

import pytest
import xarray as xr

from flexwrfinversion.loaders.footprint import LoadFootprintForTotalInCity
from flexwrfinversion.loaders.measurement import MeasurementFromFile
from flexwrfinversion.loaders.target import TargetLoaderTotalInCity

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
