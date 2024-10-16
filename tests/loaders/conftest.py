from pathlib import Path

import numpy as np
import pytest

from flexwrfinversion.loaders.footprint import (
    FlexibleFootprintLoaderAnthBio,
    FlexibleFootprintLoaderAnthBioCo,
    FlexibleFootprintLoaderTotal,
)
from flexwrfinversion.loaders.measurement import (
    FlexibleMeasurementLoaderTotal,
    FlexibleMeasurementLoaderTotalCo,
)
from flexwrfinversion.loaders.prior import (
    FlatPrior,
    FlexiblePriorLoaderTotal_ShiftToBiospheric,
)
from flexwrfinversion.loaders.prior_covariance import (
    DifferenceOfPriorToTarget,
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

EXAMPLE_DIRECTORY_0 = Path(__file__).parent.parent / "data" / "example_directory_0"


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
