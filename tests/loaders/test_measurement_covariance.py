from pathlib import Path

import numpy as np
import pytest

from flexwrfinversion.loaders.footprint import (
    LoadFootprintAnthBioCO,
    LoadFootprintForTotalInCity,
)
from flexwrfinversion.loaders.measurement import (
    MeasurementFromFile,
    MeasurementFromFileCO,
)
from flexwrfinversion.loaders.measurement_covariance import (
    ConstantNoCorrelation,
    ConstantNoCorrelationCO,
)
from flexwrfinversion.loaders.target import (
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


class Test_ConstantNoCorrelation:
    @pytest.mark.parametrize("ppm_error", (2, 4))
    def test_load_timeframe(self, measurement_from_file, ppm_error):
        measurements = measurement_from_file.measurements
        start_mtime = measurements.unstack().MTime.values[0]
        end_mtime = measurements.unstack().MTime.values[4]
        constant_no_correlation = ConstantNoCorrelation(
            measurement_loader=measurement_from_file, ppm_error=ppm_error
        )
        covariance = constant_no_correlation.load_timeframe(
            start_time=start_mtime, end_time=end_mtime
        )
        assert covariance is not None
        assert covariance.shape == (10, 10)
        assert np.allclose(covariance.max(), (ppm_error * 1e-6) ** 2, atol=0)
        assert set(covariance.dims) == {"measurement0", "measurement1"}

    def test_load_timeframe_with_co(self, measurement_from_file_co):
        measurements = measurement_from_file_co.measurements
        start_mtime = measurements.unstack().MTime.values[0]
        end_mtime = measurements.unstack().MTime.values[4]
        constant_no_correlation = ConstantNoCorrelation(
            measurement_loader=measurement_from_file_co, ppm_error=2
        )
        covariance = constant_no_correlation.load_timeframe(
            start_time=start_mtime, end_time=end_mtime
        )
        assert covariance is not None
        assert covariance.shape == (20, 20)
        assert np.allclose(covariance.max(), 4e-12, atol=0)
        assert set(covariance.dims) == {"measurement0", "measurement1"}
        assert set(covariance.unstack().species0.values) == {"CO", "CO2"}


class Test_ConstantNoCorrelationCO:
    def test_load_timeframe(self, measurement_from_file_co):
        measurements = measurement_from_file_co.measurements
        start_mtime = measurements.unstack().MTime.values[0]
        end_mtime = measurements.unstack().MTime.values[4]
        constant_no_correlation = ConstantNoCorrelationCO(
            measurement_loader=measurement_from_file_co, ppm_error=2, ppb_error=10
        )
        covariance = constant_no_correlation.load_timeframe(
            start_time=start_mtime, end_time=end_mtime
        )
        assert covariance is not None
        assert covariance.shape == (20, 20)
        assert np.allclose(covariance.max(), 4e-12, atol=0)
        assert np.allclose(
            covariance.where(covariance.species0 == "CO").max(), 1e-16, atol=0
        )
        assert set(covariance.dims) == {"measurement0", "measurement1"}
        assert set(covariance.unstack().species0.values) == {"CO", "CO2"}
