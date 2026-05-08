import pytest

try:
    import pandas as pd
    import paper_analysis as pa
except Exception as exc:
    pytest.skip(f"paper analysis dependencies are unavailable: {exc}", allow_module_level=True)


def test_preference_gap_test_passes_for_monotone_negative_fixture():
    df = pd.DataFrame(
        {
            "pref_gap": [0, 1, 2, 3, 4],
            "agreement": [5, 4, 3, 2, 1],
        }
    )

    result = pa._test1_pref_gap_agreement(df)

    assert result["passed"] is True
    assert result["r"] < 0


def test_openness_test_passes_for_monotone_positive_fixture():
    df = pd.DataFrame(
        {
            "combined_openness": [0, 1, 2, 3, 4],
            "agreement": [1, 2, 3, 4, 5],
        }
    )

    result = pa._test5_openness_agreement(df)

    assert result["passed"] is True
    assert result["r"] > 0


def test_contentiousness_test_fails_when_shared_preferences_differ_by_topic():
    df = pd.DataFrame(
        {
            "pref_gap": [0] * 9,
            "contentiousness": [1, 1, 1, 2, 2, 2, 3, 3, 3],
            "agreement": [5, 5, 4, 3, 3, 3, 1, 1, 2],
        }
    )

    result = pa._test4_contentiousness_at_shared_pref(df)

    assert result["passed"] is False
    assert result["p"] < pa.P_THRESHOLD
