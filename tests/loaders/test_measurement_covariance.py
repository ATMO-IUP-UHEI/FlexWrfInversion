from pathlib import Path

import pytest

from flexwrfinversion.loaders.footprint import LoadFootprintForTotalInCity
from flexwrfinversion.loaders.measurement import MeasurementFromFile
from flexwrfinversion.loaders.measurement_covariance import ConstantNoCorrelation
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


class Test_ConstantNoCorrelation:
    @pytest.mark.parametrize("ppm_error", (2, 4))
    def test_load_timeframe(self, measurement_from_file, ppm_error):
        measurements = measurement_from_file.measurements
        start_mtime = measurements.MTime.values[0]
        end_mtime = measurements.MTime.values[4]
        constant_no_correlation = ConstantNoCorrelation(
            measurement_loader=measurement_from_file, ppm_error=ppm_error
        )
        covariance = constant_no_correlation.load_timeframe(
            start_time=start_mtime, end_time=end_mtime
        )
        assert covariance is not None
        assert covariance.shape == (5, 5)
        assert covariance.max() == (ppm_error * 1e-6) ** 2
        assert set(covariance.dims) == {"measurement0", "measurement1"}
