import importlib.util
from pathlib import Path


def load_p10_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "31_run_p10_eurosys_main_evidence.py"
    spec = importlib.util.spec_from_file_location("p10_main_evidence", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_apply_holm_correction_is_scoped_by_analysis_family():
    module = load_p10_module()
    rows = [
        {
            "analysis_scope": "scope_a",
            "unit_id": "a1",
            "mcnemar_exact_p": "0.03",
        },
        {
            "analysis_scope": "scope_a",
            "unit_id": "a2",
            "mcnemar_exact_p": "0.04",
        },
        {
            "analysis_scope": "scope_b",
            "unit_id": "b1",
            "mcnemar_exact_p": "0.04",
        },
    ]

    module.apply_holm_correction(rows)

    by_id = {row["unit_id"]: row for row in rows}
    assert by_id["a1"]["holm_adjusted_p"] == "0.06"
    assert by_id["a2"]["holm_adjusted_p"] == "0.06"
    assert by_id["a1"]["significant_after_holm_0_05"] is False
    assert by_id["a2"]["significant_after_holm_0_05"] is False
    assert by_id["b1"]["holm_adjusted_p"] == "0.04"
    assert by_id["b1"]["significant_after_holm_0_05"] is True
