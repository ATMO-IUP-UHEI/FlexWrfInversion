import xarray as xr


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
