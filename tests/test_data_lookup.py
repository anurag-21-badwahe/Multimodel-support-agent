from app.data_lookup import lookup_by_error_code


def test_exact_match_returns_structured_evidence():
    result = lookup_by_error_code("E-104")

    assert result["found"] is True
    assert result["error_code_record"]["meaning"] == "Low coolant pressure"
    assert result["component_records"][0]["component_id"] == "V2"
    assert [row["step_number"] for row in result["troubleshooting_records"]] == ["1", "2", "3"]
    assert {source["file"] for source in result["sources"]} == {
        "error_codes.csv",
        "components.csv",
        "troubleshooting.csv",
    }


def test_unknown_code_does_not_fabricate_evidence():
    result = lookup_by_error_code("E-999")

    assert result["found"] is False
    assert result["error_code_record"] is None
    assert result["component_records"] == []
    assert result["troubleshooting_records"] == []
    assert result["sources"] == []
