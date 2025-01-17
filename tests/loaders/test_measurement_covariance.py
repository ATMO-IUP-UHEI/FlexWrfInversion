import numpy as np
import pytest

from flexwrfinversion.loaders.measurement_covariance import (
    ConstantNoCorrelation,
    ConstantNoCorrelationCO,
)


class Test_ConstantNoCorrelation:
    @pytest.mark.parametrize("ppm_error", (2, 4))
    def test_load_timeframe(self, flexible_measurement_loader_total, ppm_error):
        measurements = flexible_measurement_loader_total.measurements
        start_mtime = measurements.unstack().MTime.values[0]
        end_mtime = measurements.unstack().MTime.values[4]
        constant_no_correlation = ConstantNoCorrelation(
            measurement_loader=flexible_measurement_loader_total, ppm_error=ppm_error
        )
        covariance = constant_no_correlation.load_timeframe(
            start_time=start_mtime, end_time=end_mtime
        )
        assert covariance is not None
        assert covariance.shape == (10, 10)
        assert np.allclose(covariance.max(), (ppm_error * 1e-6) ** 2, atol=0)
        assert set(covariance.dims) == {"measurement0", "measurement1"}

    def test_load_timeframe_with_co(self, flexible_measurement_loader_total_co):
        measurements = flexible_measurement_loader_total_co.measurements
        start_mtime = measurements.unstack().MTime.values[0]
        end_mtime = measurements.unstack().MTime.values[4]
        constant_no_correlation = ConstantNoCorrelation(
            measurement_loader=flexible_measurement_loader_total_co, ppm_error=2
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
    def test_load_timeframe(self, flexible_measurement_loader_total_co):
        measurements = flexible_measurement_loader_total_co.measurements
        start_mtime = measurements.unstack().MTime.values[0]
        end_mtime = measurements.unstack().MTime.values[4]
        constant_no_correlation = ConstantNoCorrelationCO(
            measurement_loader=flexible_measurement_loader_total_co,
            ppm_error=2,
            ppb_error=10,
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
