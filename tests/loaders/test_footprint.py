import numpy as np
import xarray as xr

from flexwrfinversion.loaders.footprint import FootprintLoaderTotalAndCO2_ff


class Test_FlexibleFootprintLoaderTotal:
    def test_footprint(self, flexible_footprint_loader_total):
        footprint = flexible_footprint_loader_total.footprint
        assert footprint is not None
        assert isinstance(footprint, xr.DataArray)
        assert len(footprint.dims) == 2
        assert set(footprint.dims) == {"state", "measurement"}

    def test_footprint_keep_only_leave_out(
        self,
        flexible_footprint_loader_total_keep_only,
        flexible_footprint_loader_total_leave_out,
    ):
        for footprint in [
            flexible_footprint_loader_total_keep_only.footprint,
            flexible_footprint_loader_total_leave_out.footprint,
        ]:
            assert footprint is not None
            assert isinstance(footprint, xr.DataArray)
            assert len(footprint.dims) == 2
            assert set(footprint.dims) == {"state", "measurement"}
            assert set(footprint.MPlace.values) == {b"site00"}

    def test_footprint_times_of_day(self, flexible_footprint_loader_total_times_of_day):
        footprint = flexible_footprint_loader_total_times_of_day.footprint
        assert set(footprint.MTime.dt.hour.values) == {0, 4}

    def test_load_timeframe(self, flexible_footprint_loader_total):
        footprint = flexible_footprint_loader_total.footprint
        start_time = footprint["Time"].values[0]
        end_time = footprint["Time"].values[3]
        start_mtime = footprint["MTime"].values[0]
        end_mtime = footprint["MTime"].values[4]

        footprint_timeframe = flexible_footprint_loader_total.load_timeframe(
            start_time=start_time,
            end_time=end_time,
            start_mtime=start_mtime,
            end_mtime=end_mtime,
        )

        assert footprint_timeframe is not None
        assert isinstance(footprint_timeframe, xr.DataArray)
        assert len(footprint_timeframe.dims) == 2
        assert set(footprint_timeframe.dims) == {"state", "measurement"}


class Test_FlexibleFootprintLoaderAnthBio:
    def test_footprint(self, flexible_footprint_loader_anth_bio):
        footprint = flexible_footprint_loader_anth_bio.footprint
        assert footprint is not None
        assert isinstance(footprint, xr.DataArray)
        assert len(footprint.dims) == 2
        assert set(footprint.dims) == {"state", "measurement"}
        assert {"sector"}.issubset(set(footprint.coords.keys()))

    def test_load_timeframe(self, flexible_footprint_loader_anth_bio):
        footprint = flexible_footprint_loader_anth_bio.footprint
        start_time = footprint["Time"].values[0]
        end_time = footprint["Time"].values[3]
        start_mtime = footprint["MTime"].values[0]
        end_mtime = footprint["MTime"].values[4]

        footprint_timeframe = flexible_footprint_loader_anth_bio.load_timeframe(
            start_time=start_time,
            end_time=end_time,
            start_mtime=start_mtime,
            end_mtime=end_mtime,
        )

        assert footprint_timeframe is not None
        assert isinstance(footprint_timeframe, xr.DataArray)
        assert len(footprint_timeframe.dims) == 2
        assert set(footprint_timeframe.dims) == {"state", "measurement"}
        assert {"sector"}.issubset(set(footprint_timeframe.coords.keys()))


class Test_FlexibleFootprintLoaderAnthBioCo:
    def test_footprint(self, flexible_footprint_loader_anth_bio_co):
        footprint = flexible_footprint_loader_anth_bio_co.footprint
        assert footprint is not None
        assert isinstance(footprint, xr.DataArray)
        assert len(footprint.dims) == 2
        assert set(footprint.dims) == {"state", "measurement"}
        assert {"sector", "Time", "MTime", "subsector", "MPlace", "species"}.issubset(
            set(footprint.coords.keys())
        )

    def test_load_timeframe(self, flexible_footprint_loader_anth_bio_co):
        footprint = flexible_footprint_loader_anth_bio_co.footprint
        start_time = footprint["Time"].values[0]
        end_time = footprint["Time"].values[3]
        start_mtime = footprint["MTime"].values[0]
        end_mtime = footprint["MTime"].values[4]

        footprint_timeframe = flexible_footprint_loader_anth_bio_co.load_timeframe(
            start_time=start_time,
            end_time=end_time,
            start_mtime=start_mtime,
            end_mtime=end_mtime,
        )

        assert footprint_timeframe is not None
        assert isinstance(footprint_timeframe, xr.DataArray)
        assert len(footprint_timeframe.dims) == 2
        assert set(footprint_timeframe.dims) == {"state", "measurement"}
        assert {"sector", "Time", "MTime", "subsector", "MPlace", "species"}.issubset(
            set(footprint_timeframe.coords.keys())
        )


class Test_FootprintLoaderTotalAndCO2_ff:
    def test_footprint(
        self, footprint_loader_total_and_co2_ff: FootprintLoaderTotalAndCO2_ff
    ):
        footprint = footprint_loader_total_and_co2_ff.footprint
        original_city_footprint = xr.open_dataset(
            footprint_loader_total_and_co2_ff._footprint_file_city_ant
        )[footprint_loader_total_and_co2_ff.ant_sector_key]
        original_city_footprint_co2_ff = xr.open_dataset(
            footprint_loader_total_and_co2_ff._footprint_file_city_co2_ff
        )[footprint_loader_total_and_co2_ff._weekly_co2_ff_sector_key]
        assert footprint is not None
        assert isinstance(footprint, xr.DataArray)
        assert len(footprint.dims) == 2
        assert set(footprint.dims) == {"state", "measurement"}
        assert set(footprint.coords) == {
            "state",
            "measurement",
            "Time",
            "MTime",
            "MPlace",
            "subsector",
            "sector",
            "group",
        }
        mplace = original_city_footprint.MPlace[3].values
        mtime = original_city_footprint.MTime[2].values
        subsector = original_city_footprint.subsector[1].values
        sector = footprint_loader_total_and_co2_ff.ant_sector_key
        time = original_city_footprint.Time[0].values
        mtime_co2_ff = original_city_footprint_co2_ff.MTime[1].values
        mplace_co2_ff = FootprintLoaderTotalAndCO2_ff.CO2_FF_MPLACE_NAME.encode()

        assert np.isclose(
            original_city_footprint.sel(
                MPlace=mplace, MTime=mtime, subsector=subsector, Time=time
            ).item(),
            footprint.sel(
                measurement=((footprint.MPlace == mplace) & (footprint.MTime == mtime)),
                state=(
                    (footprint.subsector == subsector)
                    & (footprint.sector == sector)
                    & (footprint.Time == time)
                ),
            ).item(),
            atol=0,
            rtol=1e-6,
        )

        assert np.isclose(
            original_city_footprint_co2_ff.sel(
                measurement_id=original_city_footprint_co2_ff.MTime == mtime_co2_ff,
                subsector=subsector,
                Time=time,
            ).item(),
            footprint.sel(
                measurement=(
                    (footprint.MPlace == mplace_co2_ff)
                    & (footprint.MTime == mtime_co2_ff)
                ),
                state=(
                    (footprint.subsector == subsector)
                    & (footprint.sector == sector)
                    & (footprint.Time == time)
                ),
            ).item(),
            atol=0,
            rtol=1e-6,
        )
        assert (
            xr.zeros_like(
                footprint.sel(
                    measurement=(
                        (footprint.MPlace == mplace_co2_ff)
                        & (footprint.MTime == mtime_co2_ff)
                    ),
                    state=(
                        footprint.sector
                        == footprint_loader_total_and_co2_ff.ant_sector_key
                    ),
                )
            )
            == footprint.sel(
                measurement=(
                    (footprint.MPlace == mplace_co2_ff)
                    & (footprint.MTime == mtime_co2_ff)
                ),
                state=(
                    footprint.sector == footprint_loader_total_and_co2_ff.bio_sector_key
                ),
            )
        ).all()

    def test_load_timeframe(self, footprint_loader_total_and_co2_ff):
        footprint = footprint_loader_total_and_co2_ff.footprint
        start_time = footprint["Time"].values[0]
        end_time = footprint["Time"].values[3]
        start_mtime = footprint["MTime"].values[0]
        end_mtime = footprint["MTime"].values[4]

        footprint_timeframe = footprint_loader_total_and_co2_ff.load_timeframe(
            start_time=start_time,
            end_time=end_time,
            start_mtime=start_mtime,
            end_mtime=end_mtime,
        )

        assert isinstance(footprint_timeframe, xr.DataArray)
        assert len(footprint_timeframe.dims) == 2
        assert set(footprint_timeframe.dims) == {"state", "measurement"}
        assert set(footprint_timeframe.coords) == {
            "state",
            "measurement",
            "Time",
            "MTime",
            "MPlace",
            "subsector",
            "sector",
            "group",
        }
        assert (footprint_timeframe.MTime >= start_mtime).all() and (
            footprint_timeframe.MTime <= end_mtime
        ).all()
        assert (footprint_timeframe.Time >= start_time).all() and (
            footprint_timeframe.Time <= end_time
        ).all()

    def test_load_timeframe_consistency(
        self, footprint_loader_total_and_co2_ff, measurement_loader_total_and_co2_ff
    ):
        footprint = footprint_loader_total_and_co2_ff.footprint
        measurement_loader = measurement_loader_total_and_co2_ff
        start_time = footprint["Time"].values[0]
        end_time = footprint["Time"].values[-1]
        start_mtime = measurement_loader.measurements.MTime.values[0]
        end_mtime = measurement_loader.measurements.MTime.values[-20]
        measurements = measurement_loader.load_timeframe(start_mtime, end_mtime)
        footprint = footprint_loader_total_and_co2_ff.load_timeframe(
            start_time, end_time, start_mtime, end_mtime
        )
        for variable in ["MTime", "MPlace", "measurement"]:
            assert (measurements[variable].values == footprint[variable].values).all()

    def test_add_bio_sector_if_needed(self, footprint_loader_total_and_co2_ff):
        fp_loader = footprint_loader_total_and_co2_ff
        total_footprint = fp_loader._adjust_total_measurement_coords(
            super(FootprintLoaderTotalAndCO2_ff, fp_loader).footprint
        )
        co2_ff = fp_loader._adjust_co2_ff_measurement_coords(
            fp_loader._combine_subsectors(
                fp_loader._open_and_prepare(fp_loader._footprint_file_city_co2_ff)[
                    fp_loader._weekly_co2_ff_sector_key
                ],
                fp_loader._open_and_prepare(fp_loader._footprint_file_germany_co2_ff)[
                    fp_loader._weekly_co2_ff_sector_key
                ],
            ),
            total_footprint.sizes["measurement"],
            fp_loader.CO2_FF_MPLACE_NAME,
        )
        new_ff_footprint = fp_loader._add_bio_sector_if_needed(co2_ff, total_footprint)
        stacked_new_ff_footprint = new_ff_footprint.stack(state=fp_loader.STATE_DIMS)
        assert set(total_footprint.sector.values) == set(new_ff_footprint.sector.values)
        assert (
            stacked_new_ff_footprint.state.values == total_footprint.state.values
        ).all()
        assert (
            stacked_new_ff_footprint.subsector.values
            == total_footprint.subsector.values
        ).all()
        assert (
            stacked_new_ff_footprint.Time.values == total_footprint.Time.values
        ).all()
