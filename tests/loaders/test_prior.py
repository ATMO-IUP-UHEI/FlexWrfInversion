import pytest
import xarray as xr


class Test_FlatPrior:
    def test_prior(self, flat_prior, flat_prior_with_co):
        prior = flat_prior.prior
        assert prior is not None
        assert isinstance(prior, xr.DataArray)
        assert len(prior.dims) == 1
        assert set(prior.dims) == {"state"}
        assert prior.max().item() == pytest.approx(0.1, abs=1e-6)
        assert {"CO2_ANT_TOTAL", "E_CO2_VPRM"} == set(prior.sector.values)

        prior = flat_prior_with_co.prior
        assert prior is not None
        assert isinstance(prior, xr.DataArray)
        assert len(prior.dims) == 1
        assert set(prior.dims) == {"state"}
        assert prior.max().item() == pytest.approx(0.1, abs=1e-6)
        assert {"CO2_ANT_TOTAL", "E_CO2_VPRM", "E_CO"} == set(prior.sector.values)

    def test_load_timeframe(self, flat_prior, flat_prior_with_co):
        prior = flat_prior.prior
        start_time = prior["Time"].values[0]
        end_time = prior["Time"].values[3]

        prior_timeframe = flat_prior.load_timeframe(
            start_time=start_time,
            end_time=end_time,
        )

        assert prior_timeframe is not None
        assert isinstance(prior_timeframe, xr.DataArray)
        assert len(prior_timeframe.dims) == 1
        assert set(prior_timeframe.dims) == {"state"}
        assert (prior_timeframe == 0.1).all()
        assert {"CO2_ANT_TOTAL", "E_CO2_VPRM"} == set(prior.sector.values)

        prior_timeframe_with_co = flat_prior_with_co.load_timeframe(
            start_time=start_time,
            end_time=end_time,
        )
        assert prior_timeframe_with_co is not None
        assert isinstance(prior_timeframe_with_co, xr.DataArray)
        assert len(prior_timeframe_with_co.dims) == 1
        assert set(prior_timeframe_with_co.dims) == {"state"}
        assert (prior_timeframe_with_co == 0.1).all()
        assert {"CO2_ANT_TOTAL", "E_CO2_VPRM", "E_CO"} == set(
            prior_timeframe_with_co.sector.values
        )


class Test_FlexiblePriorLoaderTotal_ShiftToBiospheric:
    def test_prior(self, flexible_prior_loader_total_shift_to_biospheric):
        prior = flexible_prior_loader_total_shift_to_biospheric.prior
        assert prior is not None
        assert isinstance(prior, xr.DataArray)
        assert len(prior.dims) == 1
        assert set(prior.dims) == {"state"}

    def test_load_timeframe(self, flexible_prior_loader_total_shift_to_biospheric):
        prior = flexible_prior_loader_total_shift_to_biospheric.prior
        start_time = prior["Time"].values[0]
        end_time = prior["Time"].values[3]

        prior_timeframe = (
            flexible_prior_loader_total_shift_to_biospheric.load_timeframe(
                start_time=start_time,
                end_time=end_time,
            )
        )

        assert prior_timeframe is not None
        assert isinstance(prior_timeframe, xr.DataArray)
        assert len(prior_timeframe.dims) == 1
        assert set(prior_timeframe.dims) == {"state"}
