"""Authenticated report persistence and staff-reviewed matching API."""
import hashlib
import json
import logging
import time
from threading import Lock
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Literal
from uuid import UUID
from zoneinfo import ZoneInfo

from fastapi import Depends, FastAPI, HTTPException
from firebase_admin import auth as firebase_auth
from google.cloud import firestore
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .runtime import db, firebase_app, identity, staff
from matching_system.report_relevance import INVALID_DESCRIPTION, is_relevant_description
from google.api_core.exceptions import GoogleAPICallError
from google.auth.exceptions import GoogleAuthError
from fastapi.responses import JSONResponse

log = logging.getLogger(__name__)

CATALOG = json.loads((Path(__file__).parent / "catalog.json").read_text())
app = FastAPI(title="Strathmore Lost & Found", version="1.0.0")


class ReportInput(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    request_id: UUID
    report_type: Literal["lost", "found"]
    item_name: str = Field(min_length=1, max_length=120)
    description: str = Field(min_length=3, max_length=2000)
    category_id: str
    location_id: str
    event_date: date
    brand: str = Field(default="", max_length=100)
    model: str = Field(default="", max_length=100)
    colour: str = Field(default="", max_length=100)
    material: str = Field(default="", max_length=100)
    distinctive_features: list[str] = Field(default_factory=list, max_length=10)
    photo_reference: str = Field(default="", max_length=300)
    sealed: bool = False
    custody_location: str = Field(default="", max_length=100)

    @model_validator(mode="after")
    def sensitive_custody(self):
        if self.report_type == "found" and self.category_id in ("wallet", "documents"):
            if not self.sealed or not self.custody_location:
                raise ValueError("Sensitive items must be sealed with a custody location")
        return self

    @field_validator("category_id", "location_id")
    @classmethod
    def canonical(cls, value, info):
        section = "categories" if info.field_name == "category_id" else "locations"
        if value not in {entry["id"] for entry in CATALOG[section]}:
            raise ValueError("Choose an entry from the shared catalog")
        return value

    @field_validator("event_date")
    @classmethod
    def past_date(cls, value):
        if value > datetime.now(ZoneInfo("Africa/Nairobi")).date():
            raise ValueError("Event date cannot be in the future")
        return value

    @field_validator("distinctive_features")
    @classmethod
    def feature_lengths(cls, value):
        if any(len(item) > 200 for item in value):
            raise ValueError("Each feature must be under 200 characters")
        return value


class Decision(BaseModel):
    model_config = ConfigDict(extra="forbid")
    status: Literal["approved", "rejected", "collected"]


class ClaimInput(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    ownership_details: str = Field(min_length=10, max_length=2000)


class CategoryRuleInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    retention_days: int = Field(ge=1, le=365)
    restricted: bool = False
    requires_seal: bool = False


class RoleInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    role: Literal["student", "office_staff", "admin"]


class AuthUsers:
    """Keep privileged Auth operations behind a replaceable service boundary."""

    def by_email(self, email):
        return firebase_auth.get_user_by_email(email, app=firebase_app())

    def by_uid(self, uid):
        return firebase_auth.get_user(uid, app=firebase_app())

    def set_claims(self, uid, claims):
        firebase_auth.set_custom_user_claims(uid, claims, app=firebase_app())

    def revoke_sessions(self, uid):
        firebase_auth.revoke_refresh_tokens(uid, app=firebase_app())


def auth_users():
    return AuthUsers()


def account_summary(account):
    return {"uid": account.uid, "email": account.email,
            "email_verified": account.email_verified,
            "role": (account.custom_claims or {}).get("role", "student")}


class DeviceInput(BaseModel):
    token: str = Field(min_length=20, max_length=4096)


def collection(kind):
    if kind not in ("lost", "found"):
        raise HTTPException(404, "Unknown report type")
    return f"{kind}_reports"


def report_access(client, kind, report_id, user):
    snapshot = client.collection(collection(kind)).document(report_id).get()
    if not snapshot.exists:
        raise HTTPException(404, "Report not found")
    row = snapshot.to_dict()
    if row["owner_uid"] != user["uid"] and not staff(user):
        raise HTTPException(403, "This report belongs to another user")
    return row


def admin_only(user):
    if user["role"] != "admin":
        raise HTTPException(403, "Technical admin access required")


def public_found(row):
    return {key: row.get(key) for key in ("id", "item_name", "category_id", "location_id", "event_date", "status")}


@app.get("/v1/office/unclaimed")
def unclaimed_items(user=Depends(identity), client=Depends(db)):
    if not staff(user):
        raise HTTPException(403, "Office staff access required")
    rules = {item.id: item.to_dict() for item in client.collection("category_rules").stream()}
    now = datetime.now(timezone.utc)
    rows = []
    for item in client.collection("found_reports").limit(100).stream():
        row = item.to_dict()
        policy = rules.get(row.get("category_id", ""), {})
        created = row.get("created_at")
        days = policy.get("retention_days")
        if row.get("status") == "active" and created and days and (now - created).days >= days:
            rows.append({"id": row["id"], "item_name": row.get("item_name", "Item"),
                "category_id": row.get("category_id", ""), "retention_days": days,
                "age_days": (now - created).days})
    return {"items": rows, "policy_configured_categories": len(rules)}


@app.get("/health")
def health():
    return {"status": "ok", "service": "authenticated_reports", "matching": "separate durable worker"}


_readiness_lock = Lock()
_readiness_cache = {"until": 0.0, "ready": False}


@app.get("/ready")
def ready():
    """Read-only readiness probe; no records or credentials are exposed."""
    # Bound probe frequency so public health requests cannot cause unbounded database reads.
    with _readiness_lock:
        if time.monotonic() >= _readiness_cache["until"]:
            try:
                client = db()
                list(client.collection("matching_jobs").limit(1).stream(timeout=5))
                _readiness_cache["ready"] = True
            except (HTTPException, GoogleAPICallError, GoogleAuthError, RuntimeError, ValueError):
                log.warning("Report service readiness check failed")
                _readiness_cache["ready"] = False
            _readiness_cache["until"] = time.monotonic() + 30
        is_ready = _readiness_cache["ready"]
    if not is_ready:
        return JSONResponse(status_code=503, content={"status": "unavailable", "service": "authenticated_reports", "firestore": "unavailable"})
    return {"status": "ready", "service": "authenticated_reports", "firestore": "reachable"}


@app.exception_handler(GoogleAPICallError)
@app.exception_handler(GoogleAuthError)
async def database_unavailable(_request, _error):
    log.warning("Report service upstream request failed: %s", type(_error).__name__)
    return JSONResponse(status_code=503, content={"detail": "Report database temporarily unavailable; retry the same request."})


@app.get("/v1/me")
def me(user=Depends(identity)):
    return user


@app.get("/v1/catalog")
def catalog(user=Depends(identity)):
    return CATALOG


@app.get("/v1/admin/category-rules")
def category_rules(user=Depends(identity), client=Depends(db)):
    admin_only(user)
    rows = {item.id: item.to_dict() for item in client.collection("category_rules").stream()}
    return {"rules": [{"category_id": entry["id"], **rows.get(entry["id"], {})}
                      for entry in CATALOG["categories"]]}


@app.get("/v1/admin/accounts")
def find_account(email: str, user=Depends(identity), users=Depends(auth_users)):
    admin_only(user)
    address = email.strip().lower()
    if not address.endswith("@strathmore.edu") or len(address) > 254:
        raise HTTPException(422, "Enter an exact Strathmore email address")
    try:
        account = users.by_email(address)
    except firebase_auth.UserNotFoundError:
        raise HTTPException(404, "Account not found. The person must sign in first") from None
    return {"account": account_summary(account)}


@app.put("/v1/admin/accounts/{uid}/role")
def update_account_role(uid: str, body: RoleInput, user=Depends(identity),
                        users=Depends(auth_users), client=Depends(db)):
    admin_only(user)
    if uid == user["uid"]:
        raise HTTPException(409, "You cannot change your own role")
    try:
        account = users.by_uid(uid)
    except firebase_auth.UserNotFoundError:
        raise HTTPException(404, "Account not found") from None
    if not account.email or not account.email.lower().endswith("@strathmore.edu") or not account.email_verified:
        raise HTTPException(422, "Only verified Strathmore accounts can receive roles")
    previous = (account.custom_claims or {}).get("role", "student")
    if previous != body.role:
        claims = dict(account.custom_claims or {})
        if body.role == "student":
            claims.pop("role", None)
        else:
            claims["role"] = body.role
        users.set_claims(uid, claims)
        users.revoke_sessions(uid)
        client.collection("audit_logs").document().set({"action": "role_changed", "actor_uid": user["uid"],
            "target_uid": uid, "previous_role": previous, "new_role": body.role,
            "at": firestore.SERVER_TIMESTAMP})
    return {"account": account_summary(users.by_uid(uid))}


@app.put("/v1/admin/category-rules/{category_id}")
def update_category_rule(category_id: str, body: CategoryRuleInput, user=Depends(identity), client=Depends(db)):
    admin_only(user)
    if category_id not in {entry["id"] for entry in CATALOG["categories"]}:
        raise HTTPException(404, "Unknown category")
    if category_id in ("wallet", "documents") and (not body.restricted or not body.requires_seal):
        raise HTTPException(422, "Wallets and documents require restricted sealed custody")
    row = {"retention_days": body.retention_days, "restricted": body.restricted,
           "requires_seal": body.requires_seal, "updated_by": user["uid"],
           "updated_at": firestore.SERVER_TIMESTAMP}
    client.collection("category_rules").document(category_id).set(row)
    client.collection("audit_logs").document().set({"action": "category_rule_updated", "category_id": category_id,
        "actor_uid": user["uid"], "at": firestore.SERVER_TIMESTAMP})
    return {"category_id": category_id, **body.model_dump()}


@app.get("/v1/admin/audit")
def audit(user=Depends(identity), client=Depends(db)):
    admin_only(user)
    rows = []
    for item in client.collection("audit_logs").order_by("at", direction=firestore.Query.DESCENDING).limit(100).stream():
        row = item.to_dict()
        rows.append({key: str(row[key]) for key in
            ("action", "category_id", "actor_uid", "target_uid", "previous_role", "new_role", "at") if key in row})
    return {"events": rows}


@app.post("/v1/reports", status_code=201)
def create_report(body: ReportInput, user=Depends(identity), client=Depends(db)):
    if body.report_type == "found" and not staff(user):
        raise HTTPException(403, "Office staff must record found inventory")
    row = body.model_dump(mode="json")
    request_id = row.pop("request_id")
    report_id = hashlib.sha256(f"{user['uid']}:{request_id}".encode()).hexdigest()[:32]
    if body.photo_reference and body.photo_reference != f"report_photos/{user['uid']}/{request_id}/photo":
        raise HTTPException(422, "Photo must belong to this report request")
    digest = hashlib.sha256(json.dumps(row, sort_keys=True).encode()).hexdigest()
    row.update(id=report_id, owner_uid=user["uid"], status="active", matching_status="pending",
               input_hash=digest, created_at=firestore.SERVER_TIMESTAMP)
    ref = client.collection(collection(body.report_type)).document(report_id)
    job = client.collection("matching_jobs").document(report_id)

    @firestore.transactional
    def save(transaction):
        existing = ref.get(transaction=transaction)
        if existing.exists:
            if existing.to_dict()["input_hash"] != digest:
                raise HTTPException(409, "This request ID was already used with different data")
            return existing.to_dict()
        existing_job = job.get(transaction=transaction)
        if existing_job.exists:
            raise HTTPException(409, "This request ID was already used for another report type")
        # Validate only new reports, preserving exactly-once retry for existing reports.
        if not is_relevant_description(body.description):
            raise HTTPException(422, INVALID_DESCRIPTION)
        transaction.set(ref, row)
        transaction.set(job, {"report_id": report_id, "report_type": body.report_type,
                             "status": "pending", "attempts": 0, "lease_until": 0})
        return row

    saved = save(client.transaction())
    return {"id": report_id, "report_type": body.report_type, "status": "saved",
            "matching_status": saved["matching_status"]}


@app.get("/v1/reports")
def reports(user=Depends(identity), client=Depends(db)):
    rows = []
    for kind in ("lost", "found"):
        query = client.collection(collection(kind))
        if not staff(user):
            query = query.where(filter=firestore.FieldFilter("owner_uid", "==", user["uid"]))
        rows.extend(item.to_dict() for item in query.limit(100).stream())
    return {"reports": rows, "limit_per_type": 100}


@app.get("/v1/reports/{kind}/{report_id}/matches")
def matches(kind: str, report_id: str, user=Depends(identity), client=Depends(db)):
    report = report_access(client, kind, report_id, user)
    query = client.collection("matches").where(filter=firestore.FieldFilter(f"{kind}_id", "==", report_id))
    rows = [item.to_dict() for item in query.stream()]
    suggestions = sorted(rows, key=lambda item: item["score"], reverse=True)[:20]
    status = report["matching_status"]
    outcome = (
        ("possible_matches" if suggestions else "no_matches")
        if status == "complete" else "failed" if status == "failed" else "in_progress"
    )
    return {"matching_status": status, "matching_outcome": outcome,
            "match_count": len(suggestions) if status == "complete" else None,
            "matches": suggestions}


@app.get("/v1/matches/{match_id}/candidate")
def candidate(match_id: str, user=Depends(identity), client=Depends(db)):
    match = client.collection("matches").document(match_id).get()
    if not match.exists:
        raise HTTPException(404, "Suggestion unavailable")
    row = match.to_dict()
    if row.get("owner_uid") != user["uid"] and not staff(user):
        raise HTTPException(403, "Only the lost-report owner or office staff may view this candidate")
    found = client.collection("found_reports").document(row["found_id"]).get()
    lost = client.collection("lost_reports").document(row["lost_id"]).get()
    if not found.exists or not lost.exists:
        raise HTTPException(404, "Candidate unavailable")
    existing_claim = client.collection("claims").document(match_id).get()
    result = public_found(found.to_dict())
    result.update(match_id=match_id,
                  claim_status=existing_claim.to_dict()["status"] if existing_claim.exists else "",
                  claimable=(row["owner_uid"] == user["uid"] and not staff(user) and
                             not existing_claim.exists and found.to_dict()["status"] == "active" and
                             lost.to_dict()["status"] == "active"),
                  explanation=row.get("explanation", ""),
                  model_version=row.get("model_version", ""))
    if staff(user):
        result["lost_item_name"] = lost.to_dict().get("item_name", "Item")
        result["score"] = row.get("score")
    return {"candidate": result}


@app.post("/v1/reports/{kind}/{report_id}/retry")
def retry(kind: str, report_id: str, user=Depends(identity), client=Depends(db)):
    report_access(client, kind, report_id, user)
    ref = client.collection("matching_jobs").document(report_id)

    @firestore.transactional
    def reset(transaction):
        job = ref.get(transaction=transaction).to_dict()
        if job and job["status"] in ("pending", "running"):
            return
        transaction.set(ref, {"report_id": report_id, "report_type": kind,
                              "status": "pending", "attempts": 0, "lease_until": 0})
        transaction.update(client.collection(collection(kind)).document(report_id), {"matching_status": "pending"})
    reset(client.transaction())
    return {"status": "pending"}


@app.post("/v1/matches/{match_id}/claim", status_code=201)
def claim(match_id: str, body: ClaimInput, user=Depends(identity), client=Depends(db)):
    ref = client.collection("claims").document(match_id)

    @firestore.transactional
    def save(transaction):
        match = client.collection("matches").document(match_id).get(transaction=transaction)
        if not match.exists or match.to_dict()["owner_uid"] != user["uid"]:
            raise HTTPException(403, "Only the lost-report owner may claim this suggestion")
        data = match.to_dict()
        found = client.collection("found_reports").document(data["found_id"]).get(transaction=transaction)
        lost = client.collection("lost_reports").document(data["lost_id"]).get(transaction=transaction)
        existing = ref.get(transaction=transaction)
        if existing.exists:
            return existing.to_dict()
        if found.to_dict()["status"] != "active" or lost.to_dict()["status"] != "active":
            raise HTTPException(409, "This item is no longer available")
        row = {"id": match_id, "owner_uid": user["uid"], "lost_id": data["lost_id"],
               "found_id": data["found_id"], "status": "pending",
               "item_name": found.to_dict().get("item_name", "Item"),
               "lost_item_name": lost.to_dict().get("item_name", "Item"),
               "ownership_details": body.ownership_details, "created_at": firestore.SERVER_TIMESTAMP}
        transaction.set(ref, row)
        transaction.set(client.collection("notifications").document(f"claim_{match_id}"), {
            "owner_uid": found.to_dict()["owner_uid"], "kind": "claim",
            "message": "A new ownership request is waiting for office review.",
            "claim_id": match_id, "read": False,
            "created_at": firestore.SERVER_TIMESTAMP, "push_status": "pending"})
        # Avoid serializing a SERVER_TIMESTAMP sentinel in the immediate HTTP response.
        return {key: value for key, value in row.items() if key != "created_at"}
    return save(client.transaction())


@app.get("/v1/claims")
def claims(user=Depends(identity), client=Depends(db)):
    query = client.collection("claims")
    if not staff(user):
        query = query.where(filter=firestore.FieldFilter("owner_uid", "==", user["uid"]))
    rows = [item.to_dict() for item in query.limit(100).stream()]
    return {"claims": sorted(rows, key=lambda row: (row.get("status") != "pending",
                                                     str(row.get("created_at", ""))))}


@app.get("/v1/claims/{claim_id}")
def claim_detail(claim_id: str, user=Depends(identity), client=Depends(db)):
    snap = client.collection("claims").document(claim_id).get()
    if not snap.exists:
        raise HTTPException(404, "Claim not found")
    row = snap.to_dict()
    if row["owner_uid"] != user["uid"] and not staff(user):
        raise HTTPException(403, "This claim belongs to another user")
    lost = client.collection("lost_reports").document(row["lost_id"]).get()
    found = client.collection("found_reports").document(row["found_id"]).get()
    match = client.collection("matches").document(claim_id).get()
    if not lost.exists or not found.exists:
        raise HTTPException(404, "Related report unavailable")
    lost_row, found_row = lost.to_dict(), found.to_dict()
    result = {**row, "item_name": found_row.get("item_name", "Item"),
              "lost_item_name": lost_row.get("item_name", "Item"),
              "found_report": public_found(found_row)}
    if staff(user):
        # Private evidence is only for authenticated office staff, never technical admins
        # or other students. Use the real persisted pair and current report records.
        fields = ("id", "item_name", "description", "category_id", "location_id",
                  "event_date", "brand", "model", "colour", "material",
                  "distinctive_features", "status", "custody_location", "sealed")
        result["lost_report"] = {key: lost_row.get(key) for key in fields}
        result["found_report"] = {key: found_row.get(key) for key in fields}
        evidence = match.to_dict() if match.exists else {}
        result["match_evidence"] = {key: evidence.get(key) for key in
                                    ("score", "explanation", "workflow", "model_version")}
    return {"claim": result}


@app.post("/v1/claims/{claim_id}/decision")
def decision(claim_id: str, body: Decision, user=Depends(identity), client=Depends(db)):
    if not staff(user):
        raise HTTPException(403, "Staff review required")
    ref = client.collection("claims").document(claim_id)

    @firestore.transactional
    def review(transaction):
        snap = ref.get(transaction=transaction)
        if not snap.exists:
            raise HTTPException(404, "Claim not found")
        row = snap.to_dict()
        found_ref = client.collection("found_reports").document(row["found_id"])
        lost_ref = client.collection("lost_reports").document(row["lost_id"])
        found = found_ref.get(transaction=transaction).to_dict()
        lost = lost_ref.get(transaction=transaction).to_dict()
        if found is None or lost is None:
            raise HTTPException(404, "Claim reports are no longer available")
        allowed = {"pending": {"approved", "rejected"}, "approved": {"collected", "rejected"}}
        if body.status not in allowed.get(row["status"], set()):
            raise HTTPException(409, "Invalid claim transition")
        if body.status in ("approved", "collected"):
            if any(item.get("reserved_claim") not in (None, claim_id) or item["status"] != "active" for item in (found, lost)):
                raise HTTPException(409, "Another claim already reserves or closed these reports")
            update = {"reserved_claim": claim_id}
            if body.status == "collected":
                update["status"] = "closed"
            transaction.update(found_ref, update)
            transaction.update(lost_ref, update)
        elif row["status"] == "approved":
            transaction.update(found_ref, {"reserved_claim": firestore.DELETE_FIELD})
            transaction.update(lost_ref, {"reserved_claim": firestore.DELETE_FIELD})
        change = {"status": body.status, "reviewer_uid": user["uid"]}
        if body.status == "collected":
            change["collected_at"] = firestore.SERVER_TIMESTAMP
        transaction.update(ref, change)
        transaction.set(client.collection("audit_logs").document(), {
            "claim_id": claim_id, "action": body.status, "actor_uid": user["uid"], "at": firestore.SERVER_TIMESTAMP})
        transaction.set(client.collection("notifications").document(), {
            "owner_uid": row["owner_uid"], "kind": "claim_status", "claim_status": body.status,
            "message": {"approved": "Your claim was approved. Arrange pickup with the office.",
                        "rejected": "The office could not verify your ownership claim.",
                        "collected": "Your item handover has been recorded."}[body.status],
            "claim_id": claim_id, "read": False,
            "created_at": firestore.SERVER_TIMESTAMP, "push_status": "pending"})
    review(client.transaction())
    return {"status": body.status}


@app.get("/v1/claims/{claim_id}/receipt")
def handover_receipt(claim_id: str, user=Depends(identity), client=Depends(db)):
    snap = client.collection("claims").document(claim_id).get()
    if not snap.exists:
        raise HTTPException(404, "Claim not found")
    row = snap.to_dict()
    if row["owner_uid"] != user["uid"] and not staff(user):
        raise HTTPException(403, "Receipt belongs to another user")
    if row["status"] != "collected":
        raise HTTPException(409, "Handover has not been recorded")
    found = client.collection("found_reports").document(row["found_id"]).get()
    return {"claim_id": claim_id, "item_name": found.to_dict().get("item_name", "Item") if found.exists else "Item",
            "collected_at": str(row.get("collected_at", "")), "recorded_by": row.get("reviewer_uid", "")}


@app.get("/v1/notifications")
def notifications(user=Depends(identity), client=Depends(db)):
    query = client.collection("notifications").where(filter=firestore.FieldFilter("owner_uid", "==", user["uid"]))
    query = query.order_by("created_at", direction=firestore.Query.DESCENDING).limit(100)
    rows = []
    for item in query.stream():
        row = item.to_dict()
        rows.append({**row, "id": item.id, "read": bool(row.get("read", False))})
    return {"notifications": rows, "unread_count": sum(not row["read"] for row in rows)}


@app.post("/v1/notifications/{notification_id}/read")
def read_notification(notification_id: str, user=Depends(identity), client=Depends(db)):
    ref = client.collection("notifications").document(notification_id)
    @firestore.transactional
    def mark(transaction):
        snap = ref.get(transaction=transaction)
        if not snap.exists or snap.to_dict()["owner_uid"] != user["uid"]:
            raise HTTPException(404, "Notification unavailable")
        transaction.update(ref, {"read": True, "read_at": firestore.SERVER_TIMESTAMP})
    mark(client.transaction())
    return {"id": notification_id, "read": True}


@app.post("/v1/devices")
def register_device(body: DeviceInput, user=Depends(identity), client=Depends(db)):
    key = hashlib.sha256(body.token.encode()).hexdigest()
    client.collection("device_tokens").document(key).set({"owner_uid": user["uid"], "token": body.token})
    return {"status": "registered"}
