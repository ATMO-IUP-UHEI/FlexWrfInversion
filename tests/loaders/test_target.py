import xarray as xr


class Test_FlexibleTargetLoaderTotal:
    def test_target(self, flexible_target_loader_total):
        target = flexible_target_loader_total.target
        assert target is not None
        assert isinstance(target, xr.DataArray)
        assert len(target.dims) == 1
        assert set(target.dims) == {"state"}

    def test_load_timeframe(self, flexible_target_loader_total):
        target = flexible_target_loader_total.target
        start_time = target["Time"].values[0]
        end_time = target["Time"].values[3]

        target_timeframe = flexible_target_loader_total.load_timeframe(
            start_time=start_time,
            end_time=end_time,
        )

        assert target_timeframe is not None
        assert isinstance(target_timeframe, xr.DataArray)
        assert len(target_timeframe.dims) == 1
        assert set(target_timeframe.dims) == {"state"}


class Test_FlexibleTargetLoaderAnthBio:
    def test_target(self, flexible_target_loader_anth_bio):
        target = flexible_target_loader_anth_bio.target
        assert target is not None
        assert isinstance(target, xr.DataArray)
        assert len(target.dims) == 1
        assert set(target.dims) == {"state"}

    def test_load_timeframe(self, flexible_target_loader_anth_bio):
        target = flexible_target_loader_anth_bio.target
        start_time = target["Time"].values[0]
        end_time = target["Time"].values[3]

        target_timeframe = flexible_target_loader_anth_bio.load_timeframe(
            start_time=start_time,
            end_time=end_time,
        )

        assert target_timeframe is not None
        assert isinstance(target_timeframe, xr.DataArray)
        assert len(target_timeframe.dims) == 1
        assert set(target_timeframe.dims) == {"state"}


class Test_FlexibleTargetLoaderAnthBioCo:
    def test_target(self, flexible_target_loader_anth_bio_co):
        target = flexible_target_loader_anth_bio_co.target
        assert target is not None
        assert isinstance(target, xr.DataArray)
        assert len(target.dims) == 1
        assert set(target.dims) == {"state"}
        assert set(target.unstack().dims) == {"subsector", "sector", "Time"}

    def test_load_timeframe(self, flexible_target_loader_anth_bio_co):
        target = flexible_target_loader_anth_bio_co.target
        start_time = target["Time"].values[0]
        end_time = target["Time"].values[3]

        target_timeframe = flexible_target_loader_anth_bio_co.load_timeframe(
            start_time=start_time,
            end_time=end_time,
        )

        assert target_timeframe is not None
        assert isinstance(target_timeframe, xr.DataArray)
        assert len(target_timeframe.dims) == 1
        assert set(target_timeframe.dims) == {"state"}
