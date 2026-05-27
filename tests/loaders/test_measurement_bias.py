import numpy as np
import pytest

from flexwrfinversion.loaders.measurement_bias import (
    ConstantBias,
    ConstantBiasTotalOnly,
    ConstantPlusRandomBias,
    RandomStaticBias,
    RelativeBias,
)
from flexwrfinversion.loaders.measurement_covariance import ConstantNoCorrelation


@pytest.fixture
def covariance_constant_no_correlation(flexible_measurement_loader_total):
    return ConstantNoCorrelation(flexible_measurement_loader_total, ppm_error=2)


class Test_ConstantBias:
    def test_generate_bias(
        self, flexible_measurement_loader_total, covariance_constant_no_correlation
    ):
        constant_bias = ConstantBias(
            measurement_loader=flexible_measurement_loader_total,
            measurement_covariance_loader=covariance_constant_no_correlation,
            bias=2,
        )
        measurements = flexible_measurement_loader_total.measurements
        bias = constant_bias.generate_bias(measurements)
        assert bias.shape == measurements.shape
        assert set(bias.dims) == set(measurements.dims)
        assert np.allclose(bias, 2e-6, atol=0)
        assert np.allclose(bias + measurements, measurements + 2e-6, atol=0)


class Test_ConstantBiasTotalOnly:
    def test_generate_bias(
        self,
        measurement_loader_total_and_co2_ff,
        from_file_no_correlation_co2_ff,
    ):
        constant_bias = ConstantBiasTotalOnly(
            measurement_loader=measurement_loader_total_and_co2_ff,
            measurement_covariance_loader=from_file_no_correlation_co2_ff,
            bias=2,
        )
        measurements = measurement_loader_total_and_co2_ff.measurements
        bias = constant_bias.generate_bias(measurements)

        assert bias.shape == measurements.shape
        assert set(bias.dims) == set(measurements.dims)

        co2_ff_name = measurement_loader_total_and_co2_ff.CO2_FF_MPLACE_NAME
        if measurements.MPlace.dtype.kind == "S":
            co2_ff_name = np.array(co2_ff_name, dtype="S")

        co2_ff_mask = measurements.MPlace == co2_ff_name
        assert bool(co2_ff_mask.any())
        assert np.allclose(bias.where(co2_ff_mask, drop=True), 0.0, atol=0)
        assert np.allclose(bias.where(~co2_ff_mask, drop=True), 2e-6, atol=0)


class Test_RandomStaticBias:
    def test_set_bias(
        self, flexible_measurement_loader_total, covariance_constant_no_correlation
    ):
        random_static_bias = RandomStaticBias(
            measurement_loader=flexible_measurement_loader_total,
            measurement_covariance_loader=covariance_constant_no_correlation,
            standard_devition_ppm=2,
        )
        assert random_static_bias._bias is None
        random_static_bias.set_bias()
        bias = random_static_bias._bias
        unstacked_bias = bias.unstack()
        assert bias.shape == flexible_measurement_loader_total.measurements.shape
        bias_value = 0
        for mplace in unstacked_bias.MPlace:
            mplace_bias = unstacked_bias.sel(MPlace=mplace)
            assert (mplace_bias[0] == mplace_bias).all()
            assert mplace_bias[0] != bias_value
            bias_value = mplace_bias[0].item()

        assert bias.max() < 1e-4

    def test_generate_bias(
        self, flexible_measurement_loader_total, covariance_constant_no_correlation
    ):
        random_static_bias = RandomStaticBias(
            measurement_loader=flexible_measurement_loader_total,
            measurement_covariance_loader=covariance_constant_no_correlation,
            standard_devition_ppm=2,
        )
        random_static_bias.set_bias()
        measurements = (
            flexible_measurement_loader_total.measurements.unstack()
            .isel(MTime=slice(1, None), MPlace=slice(1, None))
            .stack(
                measurement=(
                    flexible_measurement_loader_total.footprint_loader.MEASUREMENT_DIMS
                )
            )
        )
        bias = random_static_bias.generate_bias(measurements)

        assert bias.shape == measurements.shape
        assert set(bias.dims) == set(measurements.dims)
        assert set(bias.unstack().dims) == set(measurements.unstack().dims)
        assert ((measurements + bias) != measurements).all()
        assert bias.max() < 1e-4


class Test_RelativeBias:
    def test_generate_bias(
        self, flexible_measurement_loader_total, covariance_constant_no_correlation
    ):
        relative_bias = RelativeBias(
            measurement_loader=flexible_measurement_loader_total,
            measurement_covariance_loader=covariance_constant_no_correlation,
            relative_bias=0.2,
        )
        measurements = flexible_measurement_loader_total.measurements
        bias = relative_bias.generate_bias(measurements)
        assert bias.shape == measurements.shape
        assert set(bias.dims) == set(measurements.dims)
        assert np.allclose(bias, 0.2 * measurements, atol=0)


class Test_ConstantPlusRandomBias:
    def test_generate_bias(
        self, flexible_measurement_loader_total, covariance_constant_no_correlation
    ):
        constant_plus_random_bias = ConstantPlusRandomBias(
            measurement_loader=flexible_measurement_loader_total,
            measurement_covariance_loader=covariance_constant_no_correlation,
            constant_bias=2,
            standard_devition_ppm=2,
        )
        constant_plus_random_bias.set_bias()
        measurements = flexible_measurement_loader_total.measurements
        bias0 = constant_plus_random_bias.generate_bias(measurements)
        constant_plus_random_bias.set_bias()
        bias1 = constant_plus_random_bias.generate_bias(measurements)
        assert bias0.shape == measurements.shape
        assert set(bias0.dims) == set(measurements.dims)
        assert (bias1 != bias0).all()
        assert np.allclose(bias1 - constant_plus_random_bias._bias, 0)
