"""Bulk CSV import: validation, modes, scoring and progress."""

from __future__ import annotations

import io

import pytest

from app.models.customer import Customer
from app.models.prediction import Prediction

HEADER = "external_id,name,email,tenure_months,monthly_charges,contract_type,internet_service,tech_support,is_active\n"


def _upload(client, headers, text, **form):
    data = {"file": (io.BytesIO(text.encode()), "c.csv"), **form}
    return client.post("/imports", headers=headers, data=data, content_type="multipart/form-data")


def test_template_download(client, analyst_headers):
    resp = client.get("/imports/template.csv", headers=analyst_headers)
    assert resp.status_code == 200 and resp.mimetype == "text/csv"
    assert resp.get_data(as_text=True).startswith("external_id,name,email")


def test_import_creates_scores_and_reports_rejects(app, client, admin_headers, mocked_predict):
    csv = HEADER + (
        "A1,Aarav,a1@example.com,5,70.5,month-to-month,Fiber optic,No,true\n"
        "A2,Bina,a2@example.com,40,30,Two year,DSL,Yes,true\n"
        "A3,Chet,a3@example.com,12,50,one-year,DSL,Yes,false\n"
        "A4,,bad-email,1,1,month-to-month,DSL,No,true\n"
        "A5,Dup,a1@example.com,3,10,month-to-month,DSL,No,true\n"
        "A6,Eve,a6@example.com,2,10,weekly,DSL,No,true\n"
    )
    resp = _upload(client, admin_headers, csv)
    assert resp.status_code == 202, resp.get_json()
    job = client.get(f"/imports/{resp.get_json()['id']}", headers=admin_headers).get_json()
    assert job["status"] == "completed" and job["phase"] == "done"
    assert (job["created_count"], job["rejected_count"], job["scored_count"]) == (3, 3, 2)  # churned A3 is not scored
    assert {e["row"] for e in job["errors"]} == {5, 6, 7}
    with app.app_context():
        assert Customer.query.count() == 3 and Prediction.query.count() == 2
        assert Customer.query.filter_by(external_id="A3").one().is_active is False
        assert Customer.query.filter_by(external_id="A2").one().contract_type == "two-year"


def test_skip_vs_update_mode(app, client, admin_headers):
    first = HEADER + "B1,Old Name,b1@example.com,5,70,month-to-month,DSL,No,true\n"
    second = HEADER + "B1,New Name,b1@example.com,6,80,one-year,DSL,No,true\n"
    _upload(client, admin_headers, first, score_after="false")
    skipped = _upload(client, admin_headers, second, score_after="false").get_json()
    assert skipped["skipped_count"] == 1 and skipped["updated_count"] == 0
    updated = _upload(client, admin_headers, second, mode="update", score_after="false").get_json()
    assert updated["updated_count"] == 1
    with app.app_context():
        c = Customer.query.filter_by(external_id="B1").one()
        assert (c.name, c.tenure_months, c.contract_type) == ("New Name", 6, "one-year")


def test_email_owned_by_another_customer_is_rejected(client, admin_headers):
    _upload(client, admin_headers, HEADER + "C1,A,c@example.com,1,10,month-to-month,DSL,No,true\n", score_after="false")
    job = _upload(
        client, admin_headers, HEADER + "C2,B,c@example.com,1,10,month-to-month,DSL,No,true\n", score_after="false"
    ).get_json()
    assert job["created_count"] == 0 and job["rejected_count"] == 1


@pytest.mark.parametrize(
    "text, code",
    [("", "empty_file"), ("name,email\nA,a@example.com\n", "missing_columns")],
)
def test_bad_files_are_rejected_up_front(client, admin_headers, text, code):
    resp = _upload(client, admin_headers, text)
    assert resp.status_code == 400 and resp.get_json()["error"]["code"] == code


def test_missing_file_and_permissions(client, admin_headers, analyst_headers):
    assert (
        client.post("/imports", headers=admin_headers, data={}, content_type="multipart/form-data").status_code == 422
    )
    assert _upload(client, analyst_headers, HEADER).status_code == 403  # only admins may import
    assert client.get("/imports", headers=analyst_headers).status_code == 200
    assert client.get("/imports/9999", headers=analyst_headers).status_code == 404
    assert client.get("/imports").status_code == 401
