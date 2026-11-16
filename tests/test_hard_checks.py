from trialmatch.hard_checks import check, parse_age


def test_parse_age():
    assert parse_age("18 Years") == (18.0, "year")
    assert parse_age("6 Months") == (6.0, "month")
    assert parse_age("N/A") is None
    assert parse_age(None) is None


def test_in_range_passes(diabetic_patient, fake_trials):
    elig = fake_trials.eligibility("NCT00000001")
    res = check(diabetic_patient, elig)
    assert res.passed
    assert res.checks == {"minimum_age": "pass", "maximum_age": "pass", "sex": "pass"}


def test_max_age_inclusive_and_fail(diabetic_patient, fake_trials):
    elig = fake_trials.eligibility("NCT00000001").model_copy(update={"maximum_age": "56 Years"})
    assert check(diabetic_patient, elig).passed
    elig = elig.model_copy(update={"maximum_age": "55 Years"})
    res = check(diabetic_patient, elig)
    assert not res.passed and res.checks["maximum_age"] == "fail"


def test_sex_mismatch(diabetic_patient, fake_trials):
    elig = fake_trials.eligibility("NCT00000001").model_copy(update={"sex": "MALE"})
    res = check(diabetic_patient, elig)
    assert not res.passed and res.checks["sex"] == "fail"
