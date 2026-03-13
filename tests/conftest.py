from pathlib import Path

import numpy as np
import pytest
import xarray as xr
import yaml

from flexwrfinversion.loaders.footprint import (
    FlexibleFootprintLoaderAnthBio,
    FlexibleFootprintLoaderAnthBioCo,
    FlexibleFootprintLoaderTotal,
    FootprintLoaderTotalAndCO2_ff,
)
from flexwrfinversion.loaders.measurement import (
    FlexibleMeasurementLoaderTotal,
    FlexibleMeasurementLoaderTotalCo,
    MeasurementLoaderTotalAndCO2_ff,
)
from flexwrfinversion.loaders.measurement_covariance import FromFileNoCorrelationCO2_ff
from flexwrfinversion.loaders.prior import (
    FlatPrior,
    FlexiblePriorLoaderTotal_ShiftToBiospheric,
    PriorLoaderAnthBio_RelativeError_PointExtra,
    PriorLoaderAnthBioCo_RelativeError_PointExtra,
)
from flexwrfinversion.loaders.prior_covariance import (
    DifferenceOfPriorToTarget,
    DifferenceOfPriorToTargetMinimumFromFile,
    DifferenceOfPriorToTargetWithCO_Correlation,
    RelativeError,
    TargetAsError,
    TargetAsErrorWithCO_Correlation,
)
from flexwrfinversion.loaders.target import (
    FlexibleTargetLoaderAnthBio,
    FlexibleTargetLoaderAnthBioCo,
    FlexibleTargetLoaderTotal,
)

EXAMPLE_DIRECTORY_0 = Path(__file__).parent / "data" / "example_directory_0"
EXAMPLE_DIRECTORY_1 = Path(__file__).parent / "data" / "example_directory_1"
EXAMPLE_DIRECTORY_2 = Path(__file__).parent / "data" / "example_directory_2"
EXAMPLE_CONFIG_DIR = Path(__file__).parent / "configs"


@pytest.fixture
def example_config5():
    with (EXAMPLE_CONFIG_DIR / "example_config5.yaml").open("r") as f:
        config = yaml.safe_load(f)
    return config


# %% TARGET LOADER FIXTURES
@pytest.fixture
def flexible_target_loader_total():
    return FlexibleTargetLoaderTotal(
        EXAMPLE_DIRECTORY_0
        / "remapped_data"
        / "spring"
        / "munich"
        / "true"
        / "remapped_true_emissions_sums_3H.nc",
        EXAMPLE_DIRECTORY_0
        / "remapped_data"
        / "spring"
        / "germany"
        / "true"
        / "remapped_true_emissions_sums_3H.nc",
    )


@pytest.fixture
def flexible_target_loader_anth_bio():
    return FlexibleTargetLoaderAnthBio(
        target_file_city_bio=EXAMPLE_DIRECTORY_0
        / "remapped_data"
        / "spring"
        / "munich"
        / "true"
        / "remapped_true_emissions_3H.nc",
        target_file_city_ant=EXAMPLE_DIRECTORY_0
        / "remapped_data"
        / "spring"
        / "munich"
        / "true"
        / "remapped_true_emissions_sums_3H.nc",
        target_file_germany_bio=EXAMPLE_DIRECTORY_0
        / "remapped_data"
        / "spring"
        / "germany"
        / "true"
        / "remapped_true_emissions_vprm_co_3H.nc",
        target_file_germany_ant=EXAMPLE_DIRECTORY_0
        / "remapped_data"
        / "spring"
        / "germany"
        / "true"
        / "remapped_true_emissions_sums_3H.nc",
    )


@pytest.fixture
def flexible_target_loader_anth_bio2():
    return FlexibleTargetLoaderAnthBio(
        target_file_city_bio=EXAMPLE_DIRECTORY_1
        / "remapped_data"
        / "berlin"
        / "remapped_true_emissions_vprm.nc",
        target_file_city_ant=EXAMPLE_DIRECTORY_1
        / "remapped_data"
        / "berlin"
        / "remapped_true_emissions_anth.nc",
        target_file_germany_bio=EXAMPLE_DIRECTORY_1
        / "remapped_data"
        / "germany"
        / "remapped_true_emissions_vprm.nc",
        target_file_germany_ant=EXAMPLE_DIRECTORY_1
        / "remapped_data"
        / "germany"
        / "remapped_true_emissions_anth.nc",
    )


@pytest.fixture
def flexible_target_loader_anth_bio_co():
    return FlexibleTargetLoaderAnthBioCo(
        target_file_city_bio=EXAMPLE_DIRECTORY_0
        / "remapped_data"
        / "spring"
        / "munich"
        / "true"
        / "remapped_true_emissions_3H.nc",
        target_file_city_ant=EXAMPLE_DIRECTORY_0
        / "remapped_data"
        / "spring"
        / "munich"
        / "true"
        / "remapped_true_emissions_sums_3H.nc",
        target_file_city_co=EXAMPLE_DIRECTORY_0
        / "remapped_data"
        / "spring"
        / "munich"
        / "true"
        / "remapped_true_emissions_3H.nc",
        target_file_germany_bio=EXAMPLE_DIRECTORY_0
        / "remapped_data"
        / "spring"
        / "germany"
        / "true"
        / "remapped_true_emissions_vprm_co_3H.nc",
        target_file_germany_ant=EXAMPLE_DIRECTORY_0
        / "remapped_data"
        / "spring"
        / "germany"
        / "true"
        / "remapped_true_emissions_sums_3H.nc",
        target_file_germany_co=EXAMPLE_DIRECTORY_0
        / "remapped_data"
        / "spring"
        / "germany"
        / "true"
        / "remapped_true_emissions_vprm_co_3H.nc",
    )


@pytest.fixture
def flexible_target_loader_anth_bio_co_2():
    return FlexibleTargetLoaderAnthBioCo(
        target_file_city_bio=EXAMPLE_DIRECTORY_1
        / "remapped_data"
        / "berlin"
        / "remapped_true_emissions_vprm.nc",
        target_file_city_ant=EXAMPLE_DIRECTORY_1
        / "remapped_data"
        / "berlin"
        / "remapped_true_emissions_anth.nc",
        target_file_city_co=EXAMPLE_DIRECTORY_1
        / "remapped_data"
        / "berlin"
        / "remapped_true_emissions_co.nc",
        target_file_germany_bio=EXAMPLE_DIRECTORY_1
        / "remapped_data"
        / "germany"
        / "remapped_true_emissions_vprm.nc",
        target_file_germany_ant=EXAMPLE_DIRECTORY_1
        / "remapped_data"
        / "germany"
        / "remapped_true_emissions_anth.nc",
        target_file_germany_co=EXAMPLE_DIRECTORY_1
        / "remapped_data"
        / "germany"
        / "remapped_true_emissions_co.nc",
    )


# %% PRIOR LOADER FIXTURES
@pytest.fixture
def flat_prior(flexible_target_loader_anth_bio):
    return FlatPrior(target_loader=flexible_target_loader_anth_bio, value=0.1)


@pytest.fixture
def flat_prior_with_co(flexible_target_loader_anth_bio_co):
    return FlatPrior(target_loader=flexible_target_loader_anth_bio_co, value=0.1)


@pytest.fixture
def flexible_prior_loader_total_shift_to_biospheric(flexible_target_loader_total):
    return FlexiblePriorLoaderTotal_ShiftToBiospheric(
        target_loader=flexible_target_loader_total,
        anth_emission_file_city=EXAMPLE_DIRECTORY_0
        / "remapped_data"
        / "spring"
        / "munich"
        / "true"
        / "remapped_true_emissions_sums_3H.nc",
        anth_emission_file_germany=EXAMPLE_DIRECTORY_0
        / "remapped_data"
        / "spring"
        / "germany"
        / "true"
        / "remapped_true_emissions_sums_3H.nc",
        bio_emission_file_city=EXAMPLE_DIRECTORY_0
        / "remapped_data"
        / "spring"
        / "munich"
        / "true"
        / "remapped_true_emissions_3H.nc",
        bio_emission_file_germany=EXAMPLE_DIRECTORY_0
        / "remapped_data"
        / "spring"
        / "germany"
        / "true"
        / "remapped_true_emissions_vprm_co_3H.nc",
    )


@pytest.fixture
def prior_loader_anth_bio_relative_error_point_extra(flexible_target_loader_anth_bio2):
    return PriorLoaderAnthBio_RelativeError_PointExtra(
        target_loader=flexible_target_loader_anth_bio2,
        anth_emission_file_city=EXAMPLE_DIRECTORY_1
        / "remapped_data"
        / "berlin"
        / "remapped_true_emissions_anth.nc",
        anth_emission_file_germany=EXAMPLE_DIRECTORY_1
        / "remapped_data"
        / "germany"
        / "remapped_true_emissions_anth.nc",
        bio_emission_file_city=EXAMPLE_DIRECTORY_1
        / "remapped_data"
        / "berlin"
        / "remapped_true_emissions_vprm.nc",
        bio_emission_file_germany=EXAMPLE_DIRECTORY_1
        / "remapped_data"
        / "germany"
        / "remapped_true_emissions_vprm.nc",
        point_emission_file_city=EXAMPLE_DIRECTORY_1
        / "remapped_data"
        / "berlin"
        / "remapped_true_emissions_point.nc",
        point_emission_file_germany=EXAMPLE_DIRECTORY_1
        / "remapped_data"
        / "germany"
        / "remapped_true_emissions_point.nc",
        anth_emission_error=-0.5,
        bio_emission_error=0.3,
        point_emission_error=-0.1,
    )


@pytest.fixture
def prior_loader_anth_bio_co_relative_error_point_extra(
    flexible_target_loader_anth_bio_co_2,
):
    return PriorLoaderAnthBioCo_RelativeError_PointExtra(
        target_loader=flexible_target_loader_anth_bio_co_2,
        anth_emission_file_city=EXAMPLE_DIRECTORY_1
        / "remapped_data"
        / "berlin"
        / "remapped_true_emissions_anth.nc",
        anth_emission_file_germany=EXAMPLE_DIRECTORY_1
        / "remapped_data"
        / "germany"
        / "remapped_true_emissions_anth.nc",
        bio_emission_file_city=EXAMPLE_DIRECTORY_1
        / "remapped_data"
        / "berlin"
        / "remapped_true_emissions_vprm.nc",
        bio_emission_file_germany=EXAMPLE_DIRECTORY_1
        / "remapped_data"
        / "germany"
        / "remapped_true_emissions_vprm.nc",
        point_emission_file_city=EXAMPLE_DIRECTORY_1
        / "remapped_data"
        / "berlin"
        / "remapped_true_emissions_point.nc",
        point_emission_file_germany=EXAMPLE_DIRECTORY_1
        / "remapped_data"
        / "germany"
        / "remapped_true_emissions_point.nc",
        co_emission_file_city=EXAMPLE_DIRECTORY_1
        / "remapped_data"
        / "berlin"
        / "remapped_true_emissions_co.nc",
        co_emission_file_germany=EXAMPLE_DIRECTORY_1
        / "remapped_data"
        / "germany"
        / "remapped_true_emissions_co.nc",
        anth_emission_error=-0.5,
        bio_emission_error=0.3,
        point_emission_error=-0.1,
        co_emission_error=0.2,
    )


# %% PRIOR COVARIANCE LOADER FIXTURES
@pytest.fixture
def relative_error(
    flexible_prior_loader_total_shift_to_biospheric,
):
    return RelativeError(
        prior_loader=flexible_prior_loader_total_shift_to_biospheric,
        relative_error=0.5,
    )


@pytest.fixture
def target_as_error(flat_prior):
    return TargetAsError(
        prior_loader=flat_prior,
    )


@pytest.fixture
def target_as_error_with_minimum(flat_prior):
    return TargetAsError(
        prior_loader=flat_prior,
        minimum_error=1e-7,
    )


@pytest.fixture
def difference_of_prior_to_target(flat_prior):
    return DifferenceOfPriorToTarget(
        prior_loader=flat_prior,
    )


@pytest.fixture
def target_as_error_with_co_correlation(flat_prior_with_co):
    return TargetAsErrorWithCO_Correlation(
        prior_loader=flat_prior_with_co,
        anth_co_correlation=0.5,
    )


@pytest.fixture
def target_as_error_with_co_correlation_with_minimum(flat_prior_with_co):
    return TargetAsErrorWithCO_Correlation(
        prior_loader=flat_prior_with_co,
        anth_co_correlation=0.5,
        co2_minimum_error=1e-6,
        co_minimum_error=2e-6,
    )


@pytest.fixture
def difference_of_prior_to_target_with_co_correlation(flat_prior_with_co):
    return DifferenceOfPriorToTargetWithCO_Correlation(
        prior_loader=flat_prior_with_co,
        anth_co_correlation=0.5,
    )


@pytest.fixture
def difference_of_prior_to_target_minimum_from_file(tmp_path, flat_prior):
    minimum_file = tmp_path / "minimum_error.nc"
    target = flat_prior.target_loader.target.unstack()
    prior = flat_prior.prior.unstack()
    abs_diff = np.abs(target - prior)
    (xr.ones_like(target) * abs_diff.mean().item()).to_netcdf(minimum_file)
    return DifferenceOfPriorToTargetMinimumFromFile(
        prior_loader=flat_prior,
        minimum_error_file=minimum_file,
    )


# %% FOOTPRINT LOADER FIXTURES
@pytest.fixture
def flexible_footprint_loader_total():
    return FlexibleFootprintLoaderTotal(
        EXAMPLE_DIRECTORY_0
        / "remapped_data"
        / "spring"
        / "munich"
        / "true"
        / "remapped_footprints_sums_3H.nc",
        EXAMPLE_DIRECTORY_0
        / "remapped_data"
        / "spring"
        / "germany"
        / "true"
        / "remapped_footprints_sums_3H.nc",
    )


@pytest.fixture
def flexible_footprint_loader_total_keep_only():
    return FlexibleFootprintLoaderTotal(
        EXAMPLE_DIRECTORY_0
        / "remapped_data"
        / "spring"
        / "munich"
        / "true"
        / "remapped_footprints_sums_3H.nc",
        EXAMPLE_DIRECTORY_0
        / "remapped_data"
        / "spring"
        / "germany"
        / "true"
        / "remapped_footprints_sums_3H.nc",
        keep_only=["site00"],
    )


@pytest.fixture
def flexible_footprint_loader_total_leave_out():
    return FlexibleFootprintLoaderTotal(
        EXAMPLE_DIRECTORY_0
        / "remapped_data"
        / "spring"
        / "munich"
        / "true"
        / "remapped_footprints_sums_3H.nc",
        EXAMPLE_DIRECTORY_0
        / "remapped_data"
        / "spring"
        / "germany"
        / "true"
        / "remapped_footprints_sums_3H.nc",
        leave_out=["site01"],
    )


@pytest.fixture
def flexible_footprint_loader_total_times_of_day():
    return FlexibleFootprintLoaderTotal(
        EXAMPLE_DIRECTORY_0
        / "remapped_data"
        / "spring"
        / "munich"
        / "true"
        / "remapped_footprints_sums_3H.nc",
        EXAMPLE_DIRECTORY_0
        / "remapped_data"
        / "spring"
        / "germany"
        / "true"
        / "remapped_footprints_sums_3H.nc",
        times_of_day=[0, 4],
    )


@pytest.fixture
def flexible_footprint_loader_anth_bio():
    return FlexibleFootprintLoaderAnthBio(
        footprint_file_city_bio=EXAMPLE_DIRECTORY_0
        / "remapped_data"
        / "spring"
        / "munich"
        / "true"
        / "remapped_footprints_3H.nc",
        footprint_file_city_ant=EXAMPLE_DIRECTORY_0
        / "remapped_data"
        / "spring"
        / "munich"
        / "true"
        / "remapped_footprints_sums_3H.nc",
        footprint_file_germany_bio=EXAMPLE_DIRECTORY_0
        / "remapped_data"
        / "spring"
        / "germany"
        / "true"
        / "remapped_footprints_vprm_co_3H.nc",
        footprint_file_germany_ant=EXAMPLE_DIRECTORY_0
        / "remapped_data"
        / "spring"
        / "germany"
        / "true"
        / "remapped_footprints_sums_3H.nc",
    )


@pytest.fixture
def flexible_footprint_loader_anth_bio_co():
    return FlexibleFootprintLoaderAnthBioCo(
        footprint_file_city_bio=EXAMPLE_DIRECTORY_0
        / "remapped_data"
        / "spring"
        / "munich"
        / "true"
        / "remapped_footprints_3H.nc",
        footprint_file_city_ant=EXAMPLE_DIRECTORY_0
        / "remapped_data"
        / "spring"
        / "munich"
        / "true"
        / "remapped_footprints_sums_3H.nc",
        footprint_file_city_co=EXAMPLE_DIRECTORY_0
        / "remapped_data"
        / "spring"
        / "munich"
        / "true"
        / "remapped_footprints_3H.nc",
        footprint_file_germany_bio=EXAMPLE_DIRECTORY_0
        / "remapped_data"
        / "spring"
        / "germany"
        / "true"
        / "remapped_footprints_vprm_co_3H.nc",
        footprint_file_germany_ant=EXAMPLE_DIRECTORY_0
        / "remapped_data"
        / "spring"
        / "germany"
        / "true"
        / "remapped_footprints_sums_3H.nc",
        footprint_file_germany_co=EXAMPLE_DIRECTORY_0
        / "remapped_data"
        / "spring"
        / "germany"
        / "true"
        / "remapped_footprints_vprm_co_3H.nc",
    )


@pytest.fixture
def footprint_loader_total_and_co2_ff():
    return FootprintLoaderTotalAndCO2_ff(
        footprint_file_city_bio=EXAMPLE_DIRECTORY_2
        / "remapped_data"
        / "city"
        / "footprint_bio.nc",
        footprint_file_city_ant=EXAMPLE_DIRECTORY_2
        / "remapped_data"
        / "city"
        / "footprint_ant.nc",
        footprint_file_germany_bio=EXAMPLE_DIRECTORY_2
        / "remapped_data"
        / "germany"
        / "footprint_bio.nc",
        footprint_file_germany_ant=EXAMPLE_DIRECTORY_2
        / "remapped_data"
        / "germany"
        / "footprint_ant.nc",
        footprint_file_city_co2_ff=EXAMPLE_DIRECTORY_2
        / "remapped_data"
        / "city"
        / "footprint_co2_ff.nc",
        footprint_file_germany_co2_ff=EXAMPLE_DIRECTORY_2
        / "remapped_data"
        / "germany"
        / "footprint_co2_ff.nc",
    )


# %% MEASUREMENT LOADER FIXTURES
@pytest.fixture
def flexible_measurement_loader_total(
    flexible_target_loader_total, flexible_footprint_loader_total
):
    return FlexibleMeasurementLoaderTotal(
        target_loader=flexible_target_loader_total,
        footprint_loader=flexible_footprint_loader_total,
        measurement_file_city=EXAMPLE_DIRECTORY_0
        / "remapped_data"
        / "spring"
        / "munich"
        / "true"
        / "true_concentrations_sums.nc",
        measurement_file_germany=EXAMPLE_DIRECTORY_0
        / "remapped_data"
        / "spring"
        / "germany"
        / "true"
        / "true_concentrations_sums.nc",
    )


@pytest.fixture
def flexible_measurement_loader_total_keep_only(
    flexible_target_loader_total, flexible_footprint_loader_total
):
    return FlexibleMeasurementLoaderTotal(
        target_loader=flexible_target_loader_total,
        footprint_loader=flexible_footprint_loader_total,
        measurement_file_city=EXAMPLE_DIRECTORY_0
        / "remapped_data"
        / "spring"
        / "munich"
        / "true"
        / "true_concentrations_sums.nc",
        measurement_file_germany=EXAMPLE_DIRECTORY_0
        / "remapped_data"
        / "spring"
        / "germany"
        / "true"
        / "true_concentrations_sums.nc",
        keep_only=["site00"],
    )


@pytest.fixture
def flexible_measurement_loader_total_leave_out(
    flexible_target_loader_total, flexible_footprint_loader_total
):
    return FlexibleMeasurementLoaderTotal(
        target_loader=flexible_target_loader_total,
        footprint_loader=flexible_footprint_loader_total,
        measurement_file_city=EXAMPLE_DIRECTORY_0
        / "remapped_data"
        / "spring"
        / "munich"
        / "true"
        / "true_concentrations_sums.nc",
        measurement_file_germany=EXAMPLE_DIRECTORY_0
        / "remapped_data"
        / "spring"
        / "germany"
        / "true"
        / "true_concentrations_sums.nc",
        leave_out=["site01"],
    )


@pytest.fixture
def flexible_measurement_loader_total_times_of_day(
    flexible_target_loader_total, flexible_footprint_loader_total
):
    return FlexibleMeasurementLoaderTotal(
        target_loader=flexible_target_loader_total,
        footprint_loader=flexible_footprint_loader_total,
        measurement_file_city=EXAMPLE_DIRECTORY_0
        / "remapped_data"
        / "spring"
        / "munich"
        / "true"
        / "true_concentrations_sums.nc",
        measurement_file_germany=EXAMPLE_DIRECTORY_0
        / "remapped_data"
        / "spring"
        / "germany"
        / "true"
        / "true_concentrations_sums.nc",
        times_of_day=[0, 4],
    )


@pytest.fixture
def flexible_measurement_loader_total_noise(
    flexible_target_loader_total, flexible_footprint_loader_total
):
    np.random.seed(0)
    return FlexibleMeasurementLoaderTotal(
        target_loader=flexible_target_loader_total,
        footprint_loader=flexible_footprint_loader_total,
        measurement_file_city=EXAMPLE_DIRECTORY_0
        / "remapped_data"
        / "spring"
        / "munich"
        / "true"
        / "true_concentrations_sums.nc",
        measurement_file_germany=EXAMPLE_DIRECTORY_0
        / "remapped_data"
        / "spring"
        / "germany"
        / "true"
        / "true_concentrations_sums.nc",
        ppm_noise=2,
    )


@pytest.fixture
def flexible_measurement_loader_total_co(
    flexible_target_loader_total, flexible_footprint_loader_anth_bio_co
):
    return FlexibleMeasurementLoaderTotalCo(
        target_loader=flexible_target_loader_total,
        footprint_loader=flexible_footprint_loader_anth_bio_co,
        measurement_file_city_co2=EXAMPLE_DIRECTORY_0
        / "remapped_data"
        / "spring"
        / "munich"
        / "true"
        / "true_concentrations_sums.nc",
        measurement_file_city_co=EXAMPLE_DIRECTORY_0
        / "remapped_data"
        / "spring"
        / "munich"
        / "true"
        / "true_concentrations.nc",
        measurement_file_germany_co2=EXAMPLE_DIRECTORY_0
        / "remapped_data"
        / "spring"
        / "germany"
        / "true"
        / "true_concentrations_sums.nc",
        measurement_file_germany_co=EXAMPLE_DIRECTORY_0
        / "remapped_data"
        / "spring"
        / "germany"
        / "true"
        / "true_concentrations_vprm_co.nc",
    )


@pytest.fixture
def flexible_measurement_loader_total_co_noise(
    flexible_target_loader_total, flexible_footprint_loader_anth_bio_co
):
    np.random.seed(1)
    return FlexibleMeasurementLoaderTotalCo(
        target_loader=flexible_target_loader_total,
        footprint_loader=flexible_footprint_loader_anth_bio_co,
        measurement_file_city_co2=EXAMPLE_DIRECTORY_0
        / "remapped_data"
        / "spring"
        / "munich"
        / "true"
        / "true_concentrations_sums.nc",
        measurement_file_city_co=EXAMPLE_DIRECTORY_0
        / "remapped_data"
        / "spring"
        / "munich"
        / "true"
        / "true_concentrations.nc",
        measurement_file_germany_co2=EXAMPLE_DIRECTORY_0
        / "remapped_data"
        / "spring"
        / "germany"
        / "true"
        / "true_concentrations_sums.nc",
        measurement_file_germany_co=EXAMPLE_DIRECTORY_0
        / "remapped_data"
        / "spring"
        / "germany"
        / "true"
        / "true_concentrations_vprm_co.nc",
        ppm_noise=2,
        ppb_noise=2,
    )


@pytest.fixture
def measurement_loader_total_and_co2_ff(
    flexible_target_loader_total, flexible_footprint_loader_total
):
    return MeasurementLoaderTotalAndCO2_ff(
        target_loader=flexible_target_loader_total,
        footprint_loader=flexible_footprint_loader_total,
        measurement_file_city=EXAMPLE_DIRECTORY_2
        / "remapped_data"
        / "city"
        / "total_measurements.nc",
        measurement_file_germany=EXAMPLE_DIRECTORY_2
        / "remapped_data"
        / "germany"
        / "total_measurements.nc",
        measurement_file_co2_ff=EXAMPLE_DIRECTORY_2
        / "remapped_data"
        / "city"
        / "co2_ff_measurements.nc",
    )


# %% MEASUREMENT COVARIANCE LOADER FIXTURES
@pytest.fixture
def from_file_no_correlation_co2_ff(measurement_loader_total_and_co2_ff):
    return FromFileNoCorrelationCO2_ff(
        measurement_loader=measurement_loader_total_and_co2_ff,
        std_file=EXAMPLE_DIRECTORY_2
        / "remapped_data"
        / "city"
        / "total_measurements_std.nc",
        std_file_co2_ff=EXAMPLE_DIRECTORY_2
        / "remapped_data"
        / "city"
        / "co2_ff_measurements_std.nc",
        ppm_error=2,
        ppm_error_co2_ff=4,
        add_quadratic=True,
    )
