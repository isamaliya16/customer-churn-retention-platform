"""Bulk CSV import of customers (built for 100,000+ rows).

The file is read in chunks with pandas, validated column-wise (vectorised), written with bulk INSERT/UPDATE and
optionally scored with the vectorised model path. Progress is stored in an `ImportJob` row so the dashboard can poll it.
"""

from __future__ import annotations

import hashlib
import io
import logging
import os
import threading
from datetime import date, timedelta

import pandas as pd
from flask import current_app
from sqlalchemy import insert, select, update

from app.constants import CONTRACT_TYPES, PROFILE_CHOICES, RAW_TO_CONTRACT
from app.extensions import db
from app.models.customer import EMAIL_RE, Customer
from app.models.import_job import ImportJob
from app.services.prediction_service import PredictionService
from app.utils.cache import invalidate_analytics_cache
from app.utils.errors import NotFoundAPIError, ValidationAPIError
from app.utils.time import utcnow

logger = logging.getLogger(__name__)

CHUNK_ROWS = 5000
MAX_ERRORS = 200
REQUIRED = ("email", "name", "tenure_months", "monthly_charges", "contract_type")
TEMPLATE_COLUMNS = [
    "external_id", "name", "email", "signup_date", "tenure_months", "monthly_charges", "total_charges",
    "contract_type", "plan", "is_active", "churned_at", "gender", "senior_citizen", "partner", "dependents",
    "phone_service", "multiple_lines", "internet_service", "online_security", "online_backup",
    "device_protection", "tech_support", "streaming_tv", "streaming_movies", "paperless_billing", "payment_method",
]  # fmt: skip
TEMPLATE_EXAMPLE = (
    "CUST-000001,Aarav Patel,aarav.patel@example.com,2024-01-15,21,70.35,1477.35,month-to-month,Fiber optic,true,,"
    "Male,0,Yes,No,Yes,No,Fiber optic,No,No,No,No,Yes,No,Yes,Electronic check"
)
_TRUE = {"true", "1", "yes", "y", "t", "active"}
_FALSE = {"false", "0", "no", "n", "f", "inactive", "churned"}
_lock = threading.Lock()
_running: set[int] = set()


def template_csv() -> str:
    return ",".join(TEMPLATE_COLUMNS) + "\n" + TEMPLATE_EXAMPLE + "\n"


def _norm_header(name: str) -> str:
    return str(name).strip().lower().replace(" ", "_").replace("-", "_")


class ImportService:
    # ------------------------------------------------------------------ jobs
    @staticmethod
    def get_or_404(job_id: int) -> ImportJob:
        job = db.session.get(ImportJob, job_id)
        if not job:
            raise NotFoundAPIError(f"Import job {job_id} not found.")
        return job

    @staticmethod
    def list_jobs(limit: int = 20) -> list[ImportJob]:
        return list(db.session.scalars(select(ImportJob).order_by(ImportJob.id.desc()).limit(limit)))

    def submit(self, data: bytes, filename: str, mode: str, score_after: bool, user_id: int | None) -> ImportJob:
        """Validate the header, store the upload and start processing (background thread; inline when testing)."""
        if mode not in ("skip", "update"):
            raise ValidationAPIError("mode must be 'skip' or 'update'.")
        if not data.strip():
            raise ValidationAPIError("The uploaded file is empty.", code="empty_file")
        try:
            header = pd.read_csv(io.BytesIO(data), nrows=0, dtype=str, encoding="utf-8-sig")
        except Exception as exc:
            raise ValidationAPIError(f"Could not read the file as CSV: {exc}", code="invalid_csv") from exc
        columns = {_norm_header(c) for c in header.columns}
        missing = [c for c in REQUIRED if c not in columns]
        if missing:
            raise ValidationAPIError(
                f"The CSV is missing required column(s): {', '.join(missing)}.",
                code="missing_columns",
                details={"required": list(REQUIRED), "found": sorted(columns)},
            )
        total = max(data.count(b"\n") - (0 if data.endswith(b"\n") else -1) - 1, 0)

        job = ImportJob(
            filename=os.path.basename(filename or "upload.csv")[:255],
            mode=mode,
            score_after=score_after,
            total_rows=total,
            created_by=user_id,
        )
        db.session.add(job)
        db.session.commit()

        folder = os.path.join(current_app.instance_path, "imports")
        os.makedirs(folder, exist_ok=True)
        path = os.path.join(folder, f"job_{job.id}.csv")
        with open(path, "wb") as handle:
            handle.write(data)

        if current_app.config.get("IMPORT_SYNC", current_app.config.get("TESTING", False)):
            self.run(job.id, path)
            db.session.refresh(job)
        else:
            app = current_app._get_current_object()  # noqa: SLF001 - handed to the worker thread
            threading.Thread(target=self._run_in_context, args=(app, job.id, path), daemon=True).start()
        return job

    def _run_in_context(self, app, job_id: int, path: str) -> None:
        with app.app_context():
            self.run(job_id, path)

    # ------------------------------------------------------------------ the work
    def run(self, job_id: int, path: str) -> None:
        job = db.session.get(ImportJob, job_id)
        job.status, job.phase, job.started_at = "running", "importing", utcnow()
        db.session.commit()
        new_ids: list[int] = []
        try:
            for chunk in pd.read_csv(
                path, dtype=str, keep_default_na=False, chunksize=CHUNK_ROWS, encoding="utf-8-sig"
            ):
                chunk.columns = [_norm_header(c) for c in chunk.columns]
                new_ids.extend(self._import_chunk(job, chunk))
                db.session.commit()
            if job.score_after and new_ids:
                job.phase = "scoring"
                db.session.commit()

                def progress(done: int) -> None:
                    job.scored_count = done
                    db.session.commit()

                PredictionService().score_ids(new_ids, progress=progress)
            job.status, job.phase = "completed", "done"
            job.message = (
                f"Imported {job.created_count:,} new and updated {job.updated_count:,} customers; "
                f"rejected {job.rejected_count:,}, skipped {job.skipped_count:,}."
            )
        except ValidationAPIError as exc:
            db.session.rollback()
            job = db.session.get(ImportJob, job_id)
            job.status, job.phase, job.message = "failed", "done", exc.message[:500]
        except Exception as exc:  # noqa: BLE001 - never leave a job stuck in "running"
            logger.exception("Import job %s failed", job_id)
            db.session.rollback()
            job = db.session.get(ImportJob, job_id)
            job.status, job.phase, job.message = "failed", "done", f"Unexpected error: {exc}"[:500]
        job.finished_at = utcnow()
        db.session.commit()
        invalidate_analytics_cache()
        try:
            os.remove(path)
        except OSError:  # pragma: no cover
            pass

    def _import_chunk(self, job: ImportJob, raw: pd.DataFrame) -> list[int]:
        """Validate, then insert/update one chunk. Returns ids of ACTIVE customers touched (to be scored)."""
        row0 = job.processed_rows  # rows already handled -> row numbers in messages match the CSV (header = row 1)
        raw = raw.apply(lambda col: col.str.strip())
        reasons: dict[int, str] = {}

        def reject(mask: pd.Series, message: str) -> None:
            for idx in raw.index[mask & ~raw.index.isin(list(reasons))]:
                reasons[idx] = message

        email = raw["email"].str.lower()
        reject(~email.str.match(EMAIL_RE.pattern), "invalid or missing email")
        reject(raw["name"] == "", "name is required")
        tenure = pd.to_numeric(raw["tenure_months"], errors="coerce")
        reject(tenure.isna() | (tenure < 0) | (tenure > 1200), "tenure_months must be a number between 0 and 1200")
        monthly = pd.to_numeric(raw["monthly_charges"], errors="coerce")
        reject(monthly.isna() | (monthly < 0), "monthly_charges must be a non-negative number")
        contract = raw["contract_type"].map(lambda v: RAW_TO_CONTRACT.get(v, v.lower()))
        reject(~contract.isin(CONTRACT_TYPES), f"contract_type must be one of {', '.join(CONTRACT_TYPES)}")
        reject(email.duplicated(keep="first"), "duplicate email inside the file")

        def col(name: str) -> pd.Series:
            return raw[name] if name in raw else pd.Series([""] * len(raw), index=raw.index)

        ext = col("external_id")
        ext = ext.where(ext != "", "IMP-" + email.map(lambda e: hashlib.sha1(e.encode()).hexdigest()[:10].upper()))
        reject(ext.duplicated(keep="first"), "duplicate external_id inside the file")
        total_ch = pd.to_numeric(col("total_charges"), errors="coerce")
        total_ch = total_ch.where(total_ch.notna() & (total_ch >= 0), (tenure * monthly).round(2)).fillna(0.0)
        today = date.today()
        signup = pd.to_datetime(col("signup_date"), errors="coerce")
        churned_at = pd.to_datetime(col("churned_at"), errors="coerce")
        active_text = col("is_active").str.lower()
        is_active = ~(active_text.isin(_FALSE) | (churned_at.notna() & ~active_text.isin(_TRUE)))
        plan = col("plan").where(
            col("plan") != "", col("internet_service").where(col("internet_service") != "", "Standard")
        )

        profile: dict[str, pd.Series] = {}
        warnings = 0
        for name, choices in PROFILE_CHOICES.items():
            lookup = {c.lower(): c for c in choices}
            cleaned = col(name).str.lower().map(lookup)
            warnings += int(((col(name) != "") & cleaned.isna()).sum())
            profile[name] = cleaned.where(cleaned.notna(), None)
        senior = col("senior_citizen").str.lower().map({"1": 1, "0": 0, "yes": 1, "no": 0, "true": 1, "false": 0})

        job.processed_rows += len(raw)
        bad = pd.Series(raw.index.isin(list(reasons)), index=raw.index)
        for idx, message in list(reasons.items())[: max(0, MAX_ERRORS - len(job.errors))]:
            job.errors = [*job.errors, {"row": int(row0 + raw.index.get_loc(idx) + 2), "message": message}]
        job.rejected_count += int(bad.sum())
        if warnings and not job.warnings:
            job.warnings = [
                f"Some profile values were not recognised and were left blank (e.g. {warnings} in this chunk)."
            ]

        ok = raw.index[~bad]
        if not len(ok):
            return []

        existing_by_ext = {
            row.external_id: row
            for row in db.session.execute(
                select(Customer.id, Customer.external_id, Customer.email, Customer.is_active).where(
                    Customer.external_id.in_(ext[ok].tolist())
                )
            )
        }
        taken_emails = {
            row.email: row.external_id
            for row in db.session.execute(
                select(Customer.email, Customer.external_id).where(Customer.email.in_(email[ok].tolist()))
            )
        }
        now = utcnow()
        inserts: list[dict] = []
        updates: list[dict] = []
        touched_ext: list[str] = []
        for idx in ok:
            external_id, mail = ext[idx], email[idx]
            owner = taken_emails.get(mail)
            if owner is not None and owner != external_id:
                job.rejected_count += 1
                if len(job.errors) < MAX_ERRORS:
                    job.errors = [
                        *job.errors,
                        {
                            "row": int(row0 + raw.index.get_loc(idx) + 2),
                            "message": "email already used by another customer",
                        },
                    ]
                continue
            months = int(tenure[idx])
            sdate = signup[idx].date() if pd.notna(signup[idx]) else today - timedelta(days=int(months * 30.4375))
            active = bool(is_active[idx])
            record = {
                "external_id": external_id,
                "name": raw.at[idx, "name"][:120],
                "email": mail,
                "signup_date": sdate,
                "tenure_months": months,
                "monthly_charges": float(monthly[idx]),
                "total_charges": float(total_ch[idx]),
                "contract_type": contract[idx],
                "plan": (plan[idx] or "Standard")[:64],
                "is_active": active,
                "churned_at": (
                    None if active else (churned_at[idx].to_pydatetime() if pd.notna(churned_at[idx]) else now)
                ),
                "senior_citizen": None if pd.isna(senior[idx]) else int(senior[idx]),
                **{
                    name: (None if profile[name][idx] is None or pd.isna(profile[name][idx]) else profile[name][idx])
                    for name in profile
                },
                "updated_at": now,
            }
            existing = existing_by_ext.get(external_id)
            if existing is None:
                inserts.append({**record, "created_at": now})
            elif job.mode == "update":
                updates.append({**record, "id": existing.id})
            else:
                job.skipped_count += 1
                continue
            touched_ext.append(external_id)

        if inserts:
            db.session.execute(insert(Customer), inserts)
            job.created_count += len(inserts)
        if updates:
            db.session.execute(update(Customer), updates)
            job.updated_count += len(updates)
        db.session.flush()
        if not touched_ext:
            return []
        active_ids = db.session.scalars(
            select(Customer.id).where(Customer.external_id.in_(touched_ext), Customer.is_active.is_(True))
        )
        return list(active_ids)
