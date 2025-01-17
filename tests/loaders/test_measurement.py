import numpy as np
import xarray as xr


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

    def test_measurements_times_of_day(
        self, flexible_measurement_loader_total_times_of_day
    ):
        measurements = flexible_measurement_loader_total_times_of_day.measurements
        assert set(measurements.MTime.dt.hour.values) == {0, 4}

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

    def test_load_time_frame_noise(
        self, flexible_measurement_loader_total, flexible_measurement_loader_total_noise
    ):
        measurements = flexible_measurement_loader_total.load_timeframe(
            start_time=flexible_measurement_loader_total.measurements["MTime"].values[
                0
            ],
            end_time=flexible_measurement_loader_total.measurements["MTime"].values[4],
        )
        measurements_noise = flexible_measurement_loader_total_noise.load_timeframe(
            start_time=flexible_measurement_loader_total_noise.measurements[
                "MTime"
            ].values[0],
            end_time=flexible_measurement_loader_total_noise.measurements[
                "MTime"
            ].values[4],
        )
        assert measurements is not None
        assert np.isclose(
            np.std(measurements_noise - measurements), 2e-6, atol=0.2e-6, rtol=0
        )


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

    def test_load_time_frame_noise(
        self,
        flexible_measurement_loader_total_co,
        flexible_measurement_loader_total_co_noise,
    ):
        measurements = flexible_measurement_loader_total_co.load_timeframe(
            start_time=flexible_measurement_loader_total_co.measurements[
                "MTime"
            ].values[0],
            end_time=flexible_measurement_loader_total_co.measurements["MTime"].values[
                4
            ],
        )
        measurements_noise = flexible_measurement_loader_total_co_noise.load_timeframe(
            start_time=flexible_measurement_loader_total_co_noise.measurements[
                "MTime"
            ].values[0],
            end_time=flexible_measurement_loader_total_co_noise.measurements[
                "MTime"
            ].values[4],
        )
        assert measurements is not None
        assert np.isclose(
            np.std(
                measurements.loc[{"species": "CO"}]
                - measurements_noise.loc[{"species": "CO"}]
            ),
            2e-9,
            atol=1e-9,
            rtol=0,
        )
        assert np.isclose(
            np.std(
                measurements.loc[{"species": "CO2"}]
                - measurements_noise.loc[{"species": "CO2"}]
            ),
            2e-6,
            atol=1e-6,
            rtol=0,
        )
