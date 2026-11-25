"""Deleted reports should produce recoverable outcomes, not repeated crashes."""
from unittest.mock import Mock

import pytest
from fastapi import HTTPException

from service import api, worker


def test_claim_review_missing_report_returns_404(monkeypatch):
    monkeypatch.setattr(api.firestore, "transactional", lambda function: function)
    client = Mock()
    snapshots = [
        {"found_id": "found", "lost_id": "lost", "status": "pending"},
        None,
        {"status": "active"},
    ]
    client.collection.return_value.document.return_value.get.return_value.to_dict.side_effect = snapshots
    with pytest.raises(HTTPException) as error:
        api.decision("claim", api.Decision(status="approved"), {"uid": "office", "role": "office_staff"}, client)
    assert error.value.status_code == 404
    client.transaction.return_value.update.assert_not_called()


def test_worker_missing_source_records_failure_without_updating_missing_document(monkeypatch):
    monkeypatch.setattr(worker.firestore, "transactional", lambda function: function)
    monkeypatch.setattr(worker.uuid, "uuid4", lambda: "lease")
    client, ref = Mock(), Mock()
    ref.get.return_value.to_dict.side_effect = [
        {"status": "pending", "report_type": "lost", "report_id": "missing"},
        {"lease": "lease", "attempts": 3},
    ]
    source = client.collection.return_value.document.return_value
    source.get.return_value.to_dict.return_value = None
    source.get.return_value.exists = False
    assert worker.process_job(client, ref, Mock()) is True
    updates = client.transaction.return_value.update.call_args_list
    assert all(call.args[0] is ref for call in updates)
    assert updates[-1].args[1]["status"] == "failed"
