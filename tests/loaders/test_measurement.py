import numpy as np
import xarray as xr

from flexwrfinversion.loaders.measurement import (
    MeasurementLoaderFromSingleFileTotal,
    MeasurementLoaderTotalAndCO2_ff,
)


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


class Test_MeasurementLoaderFromSingleFileTotal:
    def test_measurements(
        self, tmp_path, flexible_footprint_loader_total, flexible_target_loader_total
    ):
        measurement_sample = (
            xr.ones_like(flexible_footprint_loader_total.footprint.isel(state=0))
            .drop_vars(["state", "Time", "subsector"])
            .unstack()
        )
        measurement_sample.MTime.encoding = (
            flexible_footprint_loader_total.footprint.MTime.encoding
        )
        measurement_sample.to_netcdf(tmp_path / "measurement_sample.nc")

        measurements = MeasurementLoaderFromSingleFileTotal(
            footprint_loader=flexible_footprint_loader_total,
            target_loader=flexible_target_loader_total,
            measurement_file=tmp_path / "measurement_sample.nc",
        ).measurements
        assert measurements is not None
        assert isinstance(measurements, xr.DataArray)
        assert len(measurements.dims) == 1
        assert set(measurements.dims) == {"measurement"}
        assert (
            measurements.measurement
            == flexible_footprint_loader_total.footprint.measurement
        ).all()

    def test_measurements_keep_only_times_of_day(
        self, tmp_path, flexible_footprint_loader_total, flexible_target_loader_total
    ):
        measurement_sample = (
            xr.ones_like(flexible_footprint_loader_total.footprint.isel(state=0))
            .drop_vars(["state", "Time", "subsector"])
            .unstack()
        )
        measurement_sample.MTime.encoding = (
            flexible_footprint_loader_total.footprint.MTime.encoding
        )
        measurement_sample.to_netcdf(tmp_path / "measurement_sample.nc")

        measurements = MeasurementLoaderFromSingleFileTotal(
            footprint_loader=flexible_footprint_loader_total,
            target_loader=flexible_target_loader_total,
            measurement_file=tmp_path / "measurement_sample.nc",
            keep_only=["site00"],
            times_of_day=[0, 4],
        ).measurements
        assert set(measurements.MTime.dt.hour.values) == {0, 4}
        assert set(measurements.MPlace.values) == {b"site00"}

    def test_load_timeframe(
        self, tmp_path, flexible_footprint_loader_total, flexible_target_loader_total
    ):
        measurement_sample = (
            xr.ones_like(flexible_footprint_loader_total.footprint.isel(state=0))
            .drop_vars(["state", "Time", "subsector"])
            .unstack()
        )
        measurement_sample.MTime.encoding = (
            flexible_footprint_loader_total.footprint.MTime.encoding
        )
        measurement_sample.to_netcdf(tmp_path / "measurement_sample.nc")

        measurements = MeasurementLoaderFromSingleFileTotal(
            footprint_loader=flexible_footprint_loader_total,
            target_loader=flexible_target_loader_total,
            measurement_file=tmp_path / "measurement_sample.nc",
        ).measurements
        start_mtime = measurements["MTime"].values[0]
        end_mtime = measurements["MTime"].values[4]

        measurements_timeframe = MeasurementLoaderFromSingleFileTotal(
            footprint_loader=flexible_footprint_loader_total,
            target_loader=flexible_target_loader_total,
            measurement_file=tmp_path / "measurement_sample.nc",
        ).load_timeframe(
            start_time=start_mtime,
            end_time=end_mtime,
        )

        assert measurements_timeframe is not None
        assert isinstance(measurements_timeframe, xr.DataArray)
        assert len(measurements_timeframe.dims) == 1
        assert set(measurements_timeframe.dims) == {"measurement"}
        assert (
            measurements_timeframe
            == (
                measurement_sample.sel(MTime=slice(start_mtime, end_mtime)).stack(
                    measurement=["MTime", "MPlace"]
                )
            )
        ).all()


class Test_MeasurementLoaderTotalAndCO2_ff:
    def test_adjust_co2_ff_measurement_coords(self):
        co2_ff_measurements = xr.DataArray(
            np.zeros(3),
            dims=["measurement_id"],
            coords={
                "measurement_id": [0, 1, 2],
                "MTime": (
                    "measurement_id",
                    [
                        np.datetime64("2024-01-01T00:00:00"),
                        np.datetime64("2024-01-08T00:00:00"),
                        np.datetime64("2024-01-15T00:00:00"),
                    ],
                ),
            },
        )
        adjusted_measurements = MeasurementLoaderTotalAndCO2_ff._adjust_co2_ff_measurement_coords(  # noqa: E501
            co2_ff_measurements,
            start_measurement_id=100,
            mplace_name=MeasurementLoaderTotalAndCO2_ff.CO2_FF_MPLACE_NAME,
        )
        # test if the measurement_id is gone
        # test if MTime is now a coordinate and not an index
        # test if MPlace coordinate is added with the correct name and is not an index
        assert "measurement_id" not in adjusted_measurements.coords
        assert "MTime" in adjusted_measurements.coords
        assert "MPlace" in adjusted_measurements.coords
        assert (
            adjusted_measurements.MPlace.values[0]
            == MeasurementLoaderTotalAndCO2_ff.CO2_FF_MPLACE_NAME.encode()
        )
        assert adjusted_measurements.MTime.values[0] == np.datetime64(
            "2024-01-01T00:00:00"
        )
        assert adjusted_measurements.MTime.values[1] == np.datetime64(
            "2024-01-08T00:00:00"
        )
        assert adjusted_measurements.MTime.values[2] == np.datetime64(
            "2024-01-15T00:00:00"
        )

        # check if measurement is the index coordinate
        assert "measurement" in adjusted_measurements.indexes
        assert "MTime" not in adjusted_measurements.indexes
        assert "MPlace" not in adjusted_measurements.indexes

    def test_adjust_total_measurement_coords(self):
        mtimes = [
            np.datetime64("2024-01-01T00:00:00"),
            np.datetime64("2024-01-02T00:00:00"),
            np.datetime64("2024-01-03T00:00:00"),
        ]
        mplaces = [b"site00", b"site01", b"site02"]
        total_measurements = xr.DataArray(
            np.zeros((3, 3)),
            dims=["MTime", "MPlace"],
            coords={
                "MTime": mtimes,
                "MPlace": mplaces,
            },
        )
        total_measurements = total_measurements.stack(measurement=["MTime", "MPlace"])
        adjusted_measurements = (
            MeasurementLoaderTotalAndCO2_ff._adjust_total_measurement_coords(
                total_measurements
            )
        )
        # test if the measurement_id is gone
        # test if MTime and MPlace are now coordinates and not indexes
        assert "MTime" in adjusted_measurements.coords
        assert "MPlace" in adjusted_measurements.coords
        # check expected coordinate values
        assert set(adjusted_measurements.MTime.values) == set(mtimes)
        assert set(adjusted_measurements.MPlace.values) == set(mplaces)
        assert np.array_equal(
            adjusted_measurements.MTime.values,
            np.concatenate([[mtime] * 3 for mtime in mtimes]),
        )
        assert np.array_equal(
            adjusted_measurements.MPlace.values, np.concatenate([mplaces] * 3)
        )
        # check if measurement is the index coordinate
        assert "measurement" in adjusted_measurements.indexes
        assert "MTime" not in adjusted_measurements.indexes
        assert "MPlace" not in adjusted_measurements.indexes

    def test_measurements(
        self,
        measurement_loader_total_and_co2_ff: MeasurementLoaderTotalAndCO2_ff,
    ):
        measurements = measurement_loader_total_and_co2_ff.measurements
        original_total_measurements = (
            xr.open_dataset(measurement_loader_total_and_co2_ff._measurement_file_city)
            + xr.open_dataset(
                measurement_loader_total_and_co2_ff._measurement_file_germany
            )
        ).CO2_TOTAL
        original_co2_ff_measurements = xr.open_dataset(
            measurement_loader_total_and_co2_ff._measurement_file_co2_ff
        ).CO2_FF

        assert measurements is not None
        assert isinstance(measurements, xr.DataArray)
        assert len(measurements.dims) == 1
        assert set(measurements.dims) == {"measurement"}
        # test if the combined measurements have the correct coordinates
        assert set(measurements.MTime.values) == set(
            original_total_measurements.MTime.values
        ) | set(original_co2_ff_measurements.MTime.values)
        assert set(measurements.MPlace.values) == set(
            original_total_measurements.MPlace.values
        ) | {MeasurementLoaderTotalAndCO2_ff.CO2_FF_MPLACE_NAME.encode()}
        # check for a few values if they are findable with the coordinates of the original
        mtime = original_total_measurements.MTime.values[6]
        mplace = original_total_measurements.MPlace.values[2]
        assert np.isclose(
            measurements.sel(
                measurement=(measurements.MTime == mtime)
                & (measurements.MPlace == mplace)
            ).item(),
            original_total_measurements.sel(MTime=mtime, MPlace=mplace).item(),
            atol=0,
            rtol=1e-6,
        )
        mtime = original_co2_ff_measurements.MTime.values[1]
        mplace = MeasurementLoaderTotalAndCO2_ff.CO2_FF_MPLACE_NAME
        assert np.isclose(
            measurements.sel(
                measurement=(measurements.MTime == mtime)
                & (measurements.MPlace == mplace.encode())
            ).item(),
            original_co2_ff_measurements.sel(
                measurement_id=original_co2_ff_measurements.MTime == mtime
            ).item(),
            atol=0,
            rtol=1e-6,
        )

    def test_load_timeframe(
        self,
        measurement_loader_total_and_co2_ff: MeasurementLoaderTotalAndCO2_ff,
    ):
        measurements = measurement_loader_total_and_co2_ff.measurements
        full_timeframe = measurement_loader_total_and_co2_ff.load_timeframe(
            start_time=measurements.MTime.min(),
            end_time=measurements.MTime.max(),
        )
        assert full_timeframe is not None
        assert isinstance(full_timeframe, xr.DataArray)
        assert len(full_timeframe.dims) == 1
        assert set(full_timeframe.dims) == {"measurement"}
        assert set(full_timeframe.coords) == {"measurement", "MTime", "MPlace"}
        assert (full_timeframe.MTime == measurements.MTime).all()
        assert (full_timeframe.MPlace == measurements.MPlace).all()
        assert (full_timeframe.measurement == measurements.measurement).all()
        assert (full_timeframe.values == measurements.values).all()

        # test a subset of the timeframe
        start_time = measurements.MTime.values[5]
        end_time = measurements.MTime.values[10]
        subset_timeframe = measurement_loader_total_and_co2_ff.load_timeframe(
            start_time=start_time,
            end_time=end_time,
        )
        assert subset_timeframe is not None
        assert isinstance(subset_timeframe, xr.DataArray)
        assert len(subset_timeframe.dims) == 1
        assert set(subset_timeframe.dims) == {"measurement"}
        assert set(subset_timeframe.coords) == {"measurement", "MTime", "MPlace"}
        assert (subset_timeframe.MTime >= start_time).all()
        assert (subset_timeframe.MTime <= end_time).all()
        assert (subset_timeframe.measurement.isin(measurements.measurement)).all()
        # check values for a few coordinates
        mtime = measurements.MTime.values[6]
        mplace = measurements.MPlace.values[2]
        assert np.isclose(
            subset_timeframe.sel(
                measurement=(subset_timeframe.MTime == mtime)
                & (subset_timeframe.MPlace == mplace)
            ).item(),
            measurements.sel(
                measurement=(measurements.MTime == mtime)
                & (measurements.MPlace == mplace)
            ).item(),
            atol=0,
            rtol=1e-6,
        )
