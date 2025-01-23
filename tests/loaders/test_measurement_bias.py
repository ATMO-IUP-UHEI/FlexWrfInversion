import numpy as np
import pytest

from flexwrfinversion.loaders.measurement_bias import (
    ConstantBias,
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
