"""CSV export safety: spreadsheet formula injection must be neutralised."""

import pytest

from app.services.export_service import rows_to_csv, safe_cell


@pytest.mark.parametrize("danger", ["=1+1", "+cmd|' /C calc'!A0", "-2+3", "@SUM(A1)", '=HYPERLINK("http://x","y")'])
def test_formula_prefixes_are_quoted(danger):
    assert safe_cell(danger) == "'" + danger


@pytest.mark.parametrize("value", ["Alice", "a=b", "", None, 42, -3.5, True])
def test_normal_values_untouched(value):
    assert safe_cell(value) == value


def test_rows_to_csv_escapes_cells():
    out = "".join(rows_to_csv(["name"], [["=EVIL()"], ["fine"]]))
    assert "'=EVIL()" in out and "fine" in out


def test_customer_export_endpoint_escapes_malicious_name(client, make_customer, admin_headers):
    make_customer(name='=HYPERLINK("http://evil.example","click")')
    body = client.get("/export/customers.csv", headers=admin_headers).get_data(as_text=True)
    assert "'=HYPERLINK" in body and ",=HYPERLINK" not in body and '"=HYPERLINK' not in body
