import numpy as np
import pytest
import xarray as xr

from flexwrfinversion.loaders.measurement_covariance import (
    ConstantNoCorrelation,
    ConstantNoCorrelationCO,
    ConstantPlusRelativeNoCorrelation,
    FromFileNoCorrelation,
    FromFileNoCorrelationCO,
    FromFileNoCorrelationCO2_ff,
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


class Test_FromFileNoCorrelation:
    def test_std(self, tmp_path, flexible_measurement_loader_total):
        std = (
            np.abs(flexible_measurement_loader_total.measurements.unstack()) * 0.1
            + 2e-6
        )
        std.to_netcdf(tmp_path / "measurement_std.nc")
        measurement_covariance = FromFileNoCorrelation(
            measurement_loader=flexible_measurement_loader_total,
            std_file=tmp_path / "measurement_std.nc",
        )
        std_loaded = measurement_covariance.std
        assert (std_loaded.unstack() == std).all()
        assert set(std_loaded.dims) == {"measurement"}
        assert (
            std_loaded.measurement
            == flexible_measurement_loader_total.measurements.measurement
        ).all()

    def test_std_ppm_error_not_quardatic(
        self, tmp_path, flexible_measurement_loader_total
    ):
        std = np.abs(flexible_measurement_loader_total.measurements.unstack()) * 0.1
        std.to_netcdf(tmp_path / "measurement_std.nc")
        measurement_covariance = FromFileNoCorrelation(
            measurement_loader=flexible_measurement_loader_total,
            std_file=tmp_path / "measurement_std.nc",
            ppm_error=2,
            add_quadratic=False,
        )
        std_loaded = measurement_covariance.std
        assert (std_loaded.unstack() == std + 2e-6).all()
        assert set(std_loaded.dims) == {"measurement"}
        assert (
            std_loaded.measurement
            == flexible_measurement_loader_total.measurements.measurement
        ).all()

    def test_std_ppm_error_quardatic(self, tmp_path, flexible_measurement_loader_total):
        std = np.abs(flexible_measurement_loader_total.measurements.unstack()) * 0.1
        std.to_netcdf(tmp_path / "measurement_std.nc")
        measurement_covariance = FromFileNoCorrelation(
            measurement_loader=flexible_measurement_loader_total,
            std_file=tmp_path / "measurement_std.nc",
            ppm_error=2,
        )
        std_loaded = measurement_covariance.std
        assert (std_loaded.unstack() > std).all()
        assert (std_loaded.unstack() == np.sqrt(std**2 + 2e-6**2)).all()
        assert set(std_loaded.dims) == {"measurement"}
        assert (
            std_loaded.measurement
            == flexible_measurement_loader_total.measurements.measurement
        ).all()

    def test_load_timeframe(self, tmp_path, flexible_measurement_loader_total):
        std = (
            np.abs(flexible_measurement_loader_total.measurements.unstack()) * 0.1
            + 2e-6
        )
        std.to_netcdf(tmp_path / "measurement_std.nc")
        measurement_covariance = FromFileNoCorrelation(
            measurement_loader=flexible_measurement_loader_total,
            std_file=tmp_path / "measurement_std.nc",
        )
        start_mtime = (
            flexible_measurement_loader_total.measurements.unstack().MTime.values[0]
        )
        end_mtime = (
            flexible_measurement_loader_total.measurements.unstack().MTime.values[4]
        )
        covariance = measurement_covariance.load_timeframe(
            start_time=start_mtime, end_time=end_mtime
        )
        assert covariance is not None
        assert covariance.shape == (10, 10)
        assert covariance.MTime0.values.min() == start_mtime
        assert covariance.MTime0.values.max() == end_mtime


class Test_FromFileNoCorrelationCO:
    def test_std(self, tmp_path, flexible_measurement_loader_total_co):
        std = (
            np.abs(flexible_measurement_loader_total_co.measurements.unstack()) * 0.1
            + 2e-6
        )
        std.to_netcdf(tmp_path / "measurement_std.nc")
        measurement_covariance = FromFileNoCorrelationCO(
            measurement_loader=flexible_measurement_loader_total_co,
            std_file=tmp_path / "measurement_std.nc",
        )
        std_loaded = measurement_covariance.std
        assert (std_loaded.unstack() == std).all()
        assert set(std_loaded.dims) == {"measurement"}
        assert (
            std_loaded.measurement
            == flexible_measurement_loader_total_co.measurements.measurement
        ).all()

    def test_std_ppm_ppb_error_not_quardatic(
        self, tmp_path, flexible_measurement_loader_total_co
    ):
        std = np.abs(flexible_measurement_loader_total_co.measurements.unstack()) * 0.1
        std.to_netcdf(tmp_path / "measurement_std.nc")
        measurement_covariance = FromFileNoCorrelationCO(
            measurement_loader=flexible_measurement_loader_total_co,
            std_file=tmp_path / "measurement_std.nc",
            ppm_error=2,
            ppb_error=10,
            add_quadratic=False,
        )
        std_loaded = measurement_covariance.std
        assert (
            std_loaded.unstack().sel(species="CO2") == std.sel(species="CO2") + 2e-6
        ).all()
        assert (
            std_loaded.unstack().sel(species="CO") == std.sel(species="CO") + 10e-9
        ).all()
        assert set(std_loaded.dims) == {"measurement"}
        assert (
            std_loaded.measurement
            == flexible_measurement_loader_total_co.measurements.measurement
        ).all()

    def test_std_ppm_ppb_error_quardatic(
        self, tmp_path, flexible_measurement_loader_total_co
    ):
        std = np.abs(flexible_measurement_loader_total_co.measurements.unstack()) * 0.1
        std.to_netcdf(tmp_path / "measurement_std.nc")
        measurement_covariance = FromFileNoCorrelationCO(
            measurement_loader=flexible_measurement_loader_total_co,
            std_file=tmp_path / "measurement_std.nc",
            ppm_error=2,
            ppb_error=10,
        )
        std_loaded = measurement_covariance.std
        assert (std_loaded.unstack() > std).all()
        assert (
            std_loaded.unstack().sel(species="CO2")
            == np.sqrt(std.sel(species="CO2") ** 2 + 2e-6**2)
        ).all()
        assert (
            std_loaded.unstack().sel(species="CO")
            == np.sqrt(std.sel(species="CO") ** 2 + 10e-9**2)
        ).all()
        assert set(std_loaded.dims) == {"measurement"}
        assert (
            std_loaded.measurement
            == flexible_measurement_loader_total_co.measurements.measurement
        ).all()

    def test_load_timeframe(self, tmp_path, flexible_measurement_loader_total_co):
        std = (
            np.abs(flexible_measurement_loader_total_co.measurements.unstack()) * 0.1
            + 2e-6
        )
        std.to_netcdf(tmp_path / "measurement_std.nc")
        measurement_covariance = FromFileNoCorrelationCO(
            measurement_loader=flexible_measurement_loader_total_co,
            std_file=tmp_path / "measurement_std.nc",
        )
        start_mtime = (
            flexible_measurement_loader_total_co.measurements.unstack().MTime.values[0]
        )
        end_mtime = (
            flexible_measurement_loader_total_co.measurements.unstack().MTime.values[4]
        )
        covariance = measurement_covariance.load_timeframe(
            start_time=start_mtime, end_time=end_mtime
        )
        assert covariance is not None
        assert covariance.shape == (20, 20)
        assert covariance.MTime0.values.min() == start_mtime
        assert covariance.MTime0.values.max() == end_mtime
        assert set(covariance.unstack().species0.values) == {"CO", "CO2"}
        assert set(covariance.dims) == {"measurement0", "measurement1"}


class Test_FromFileNoCorrelationCO2_ff:
    def test_std(self, from_file_no_correlation_co2_ff: FromFileNoCorrelationCO2_ff):
        std = from_file_no_correlation_co2_ff.std
        original_ppm_error = from_file_no_correlation_co2_ff._ppm_error * 1e-6
        original_ppm_error_co2_ff = (
            from_file_no_correlation_co2_ff._ppm_error_co2_ff * 1e-6
        )
        original_std_total = np.sqrt(
            xr.open_dataset(from_file_no_correlation_co2_ff._std_file).CO2_TOTAL ** 2
            + original_ppm_error**2
        )
        original_std_co2_ff = np.sqrt(
            xr.open_dataset(from_file_no_correlation_co2_ff._std_file_co2_ff).CO2_FF
            ** 2
            + original_ppm_error_co2_ff**2
        )

        assert set(std.dims) == {"measurement"}
        assert set(std.coords) == {"measurement", "MTime", "MPlace"}

        mtime = original_std_total.MTime[3]
        mplace = original_std_total.MPlace[4]
        mtime_co2_ff = original_std_co2_ff.MTime[1]
        mplace_co2_ff = (
            from_file_no_correlation_co2_ff.measurement_loader.CO2_FF_MPLACE_NAME.encode()
        )
        assert np.isclose(
            std.sel(measurement=(std.MTime == mtime) & (std.MPlace == mplace)).item(),
            original_std_total.sel(MTime=mtime, MPlace=mplace).item(),
            atol=0,
            rtol=1e-6,
        )
        assert np.isclose(
            std.sel(
                measurement=(std.MTime == mtime_co2_ff) & (std.MPlace == mplace_co2_ff)
            ).item(),
            original_std_co2_ff.sel(
                measurement_id=original_std_co2_ff.MTime == mtime_co2_ff
            ).item(),
            atol=0,
            rtol=1e-6,
        )

    def test_load_timeframe(
        self, from_file_no_correlation_co2_ff: FromFileNoCorrelationCO2_ff
    ):
        std = from_file_no_correlation_co2_ff.std
        full_timeframe = from_file_no_correlation_co2_ff.load_timeframe(
            std.MTime.min().values, std.MTime.max().values
        )
        part_timeframe = from_file_no_correlation_co2_ff.load_timeframe(
            std.MTime.values[-2],  # first of the co2_ff measurements
            std.MTime.values[-1] - np.timedelta64(1, "ns"),
        )
        assert set(full_timeframe.dims) == {"measurement0", "measurement1"}
        assert set(full_timeframe.coords) == {
            "measurement0",
            "measurement1",
            "MPlace0",
            "MPlace1",
            "MTime0",
            "MTime1",
        }
        assert (np.diag(full_timeframe.values) == std**2).all()
        assert std.MTime.values[-1] not in part_timeframe.MTime0.values
        assert (
            part_timeframe.sizes["measurement0"] == part_timeframe.sizes["measurement1"]
        )
        assert (
            part_timeframe.sizes["measurement0"] < full_timeframe.sizes["measurement0"]
        )
        assert set(np.diag(part_timeframe.values)).issubset(
            np.diag(full_timeframe.values)
        )

    def test_load_timeframe_consistency(
        self, from_file_no_correlation_co2_ff: FromFileNoCorrelationCO2_ff
    ):
        measurement_loader = from_file_no_correlation_co2_ff.measurement_loader
        start_mtime = measurement_loader.measurements.MTime.values[0]
        end_mtime = measurement_loader.measurements.MTime.values[-20]
        measurements = measurement_loader.load_timeframe(start_mtime, end_mtime)
        covariance = from_file_no_correlation_co2_ff.load_timeframe(
            start_mtime, end_mtime
        )
        for i in range(2):
            for variable in ["MTime", "MPlace", "measurement"]:
                assert (
                    measurements[variable].values == covariance[f"{variable}{i}"].values
                ).all()
