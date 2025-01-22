import numpy as np
import pytest
import xarray as xr

from flexwrfinversion.loaders.measurement_covariance import (
    ConstantNoCorrelation,
    ConstantNoCorrelationCO,
    ConstantPlusRelativeNoCorrelation,
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

    def test_get_measurement_subset(self, flexible_measurement_loader_total):
        constant_no_correlation = ConstantNoCorrelation(
            measurement_loader=flexible_measurement_loader_total, ppm_error=2
        )
        start_mtime = (
            flexible_measurement_loader_total.measurements.unstack().MTime.values[0]
        )
        end_mtime = (
            flexible_measurement_loader_total.measurements.unstack().MTime.values[4]
        )
        measurement_subset = constant_no_correlation._get_measurement_subset(
            start_time=start_mtime, end_time=end_mtime
        )
        assert measurement_subset is not None
        assert measurement_subset.shape == (10,)
        assert set(measurement_subset.dims) == {"measurement"}
        assert measurement_subset.MTime.values.min() == start_mtime
        assert measurement_subset.MTime.values.max() == end_mtime

    def test_set_constant_std(self, flexible_measurement_loader_total):
        constant_no_correlation = ConstantNoCorrelation(
            measurement_loader=flexible_measurement_loader_total, ppm_error=2
        )
        start_mtime = (
            flexible_measurement_loader_total.measurements.unstack().MTime.values[0]
        )
        end_mtime = (
            flexible_measurement_loader_total.measurements.unstack().MTime.values[4]
        )
        measurement_subset = constant_no_correlation._get_measurement_subset(
            start_time=start_mtime, end_time=end_mtime
        )
        std = constant_no_correlation._set_constant_std(measurement_subset)
        assert std is not None
        assert std.shape == (10,)
        assert set(std.dims) == {"measurement"}
        assert np.allclose(std.max(), 2e-6, atol=0)

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


class Test_ConstantPlusRelativeNoCorrelation:
    def test_load_timeframe(self, flexible_measurement_loader_total):
        measurements = flexible_measurement_loader_total.measurements
        start_mtime = measurements.unstack().MTime.values[0]
        end_mtime = measurements.unstack().MTime.values[4]
        constant_plus_relative_no_correlation = ConstantPlusRelativeNoCorrelation(
            measurement_loader=flexible_measurement_loader_total,
            ppm_error=2,
            relative_error=0.1,
        )
        covariance = constant_plus_relative_no_correlation.load_timeframe(
            start_time=start_mtime, end_time=end_mtime
        )
        expected_constant_covariance = ConstantNoCorrelation(
            measurement_loader=flexible_measurement_loader_total, ppm_error=2
        ).load_timeframe(start_time=start_mtime, end_time=end_mtime)

        expected_relative_covariance = xr.DataArray(
            np.diag(
                constant_plus_relative_no_correlation._get_measurement_subset(
                    start_time=start_mtime, end_time=end_mtime
                )
                * 0.1
            )
            ** 2,
            coords=expected_constant_covariance.coords,
        )

        assert covariance is not None
        assert covariance.shape == (10, 10)
        assert set(covariance.dims) == {"measurement0", "measurement1"}
        assert np.allclose(
            covariance,
            expected_constant_covariance + expected_relative_covariance,
            atol=0,
        )


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
