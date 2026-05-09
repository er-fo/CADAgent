from datetime import datetime, timedelta, timezone

import pytest

from scripts import prod_error_monitor as monitor


NOW = datetime(2026, 5, 10, 12, 0, tzinfo=timezone.utc)


def event(message, minutes_ago=0, group="/aws/ec2/cadagent", stream="service"):
    occurred_at = NOW - timedelta(minutes=minutes_ago)
    return {
        "timestamp": int(occurred_at.timestamp() * 1000),
        "message": message,
        "logGroupName": group,
        "logStreamName": stream,
    }


def test_redacts_sensitive_values_from_issue_samples():
    text = (
        "User erik@example.com failed with token=sk-test-1234567890abcdef "
        "Authorization: Bearer eyJhbGciOi.fake.jwt"
    )

    redacted = monitor.redact_sensitive_text(text)

    assert "erik@example.com" not in redacted
    assert "sk-test" not in redacted
    assert "Bearer eyJ" not in redacted
    assert "[email]" in redacted
    assert "[secret]" in redacted


def test_fingerprint_normalizes_variable_ids_and_keeps_error_context():
    first = monitor.build_fingerprint(
        event(
            "2026-05-10 11:59:00,123 - backend.main - ERROR - "
            "Session 11111111-1111-1111-1111-111111111111 failed on /composer/intake: "
            "ValueError: bad profile 8472"
        )
    )
    second = monitor.build_fingerprint(
        event(
            "2026-05-10 12:00:00,999 - backend.main - ERROR - "
            "Session 22222222-2222-2222-2222-222222222222 failed on /composer/intake: "
            "ValueError: bad profile 9999"
        )
    )

    assert first.key == second.key
    assert first.severity == "ERROR"
    assert first.exception_type == "ValueError"
    assert "/composer/intake" in first.normalized_message


def test_analyzer_opens_for_single_new_error_and_includes_related_updates():
    config = monitor.MonitorConfig()
    current = [
        event("2026-05-10 12:00:00,000 - backend.main - ERROR - RuntimeError: failed export")
    ]

    anomalies = monitor.detect_anomalies(
        current_events=current,
        prior_24h_events=[],
        same_hour_events=[],
        related_updates=[
            monitor.ProductionUpdate(
                source="codedeploy",
                identifier="d-ABC123",
                deployed_at=NOW - timedelta(hours=2),
                sha="abc123",
                url="https://github.com/er-fo/cadagent-backend-legacy/actions/runs/1",
                title="Deploy backend legacy to EC2",
            )
        ],
        now=NOW,
        config=config,
    )

    assert len(anomalies) == 1
    assert anomalies[0].classification == "new_error"
    assert anomalies[0].confidence == "deployment-correlated regression candidate"
    assert anomalies[0].related_updates[0].identifier == "d-ABC123"


def test_related_updates_are_limited_to_36_hours_before_first_seen():
    config = monitor.MonitorConfig()
    current = [
        event("2026-05-10 12:00:00,000 - backend.main - ERROR - RuntimeError: failed export")
    ]

    anomalies = monitor.detect_anomalies(
        current_events=current,
        prior_24h_events=[],
        same_hour_events=[],
        related_updates=[
            monitor.ProductionUpdate("codedeploy", "too-old", NOW - timedelta(hours=40)),
            monitor.ProductionUpdate("codedeploy", "related", NOW - timedelta(hours=35)),
            monitor.ProductionUpdate("codedeploy", "too-new", NOW + timedelta(minutes=1)),
        ],
        now=NOW,
        config=config,
    )

    assert [item.identifier for item in anomalies[0].related_updates] == ["related"]


def test_github_workflow_runs_are_converted_to_production_updates():
    updates = monitor.production_updates_from_github_runs(
        {
            "workflow_runs": [
                {
                    "id": 123,
                    "head_sha": "abcdef123456",
                    "html_url": "https://github.com/er-fo/cadagent-backend-legacy/actions/runs/123",
                    "display_title": "Deploy backend legacy to EC2",
                    "updated_at": "2026-05-10T10:30:00Z",
                }
            ]
        }
    )

    assert updates == [
        monitor.ProductionUpdate(
            source="github-actions",
            identifier="123",
            deployed_at=datetime(2026, 5, 10, 10, 30, tzinfo=timezone.utc),
            sha="abcdef123456",
            url="https://github.com/er-fo/cadagent-backend-legacy/actions/runs/123",
            title="Deploy backend legacy to EC2",
        )
    ]


def test_warning_fingerprint_requires_three_new_occurrences():
    config = monitor.MonitorConfig()

    two_warnings = [
        event("2026-05-10 12:00:00,000 - backend.main - WARNING - Router fallback used", 1),
        event("2026-05-10 12:01:00,000 - backend.main - WARNING - Router fallback used", 2),
    ]
    three_warnings = two_warnings + [
        event("2026-05-10 12:02:00,000 - backend.main - WARNING - Router fallback used", 3)
    ]

    assert (
        monitor.detect_anomalies(
            current_events=two_warnings,
            prior_24h_events=[],
            same_hour_events=[],
            related_updates=[],
            now=NOW,
            config=config,
        )
        == []
    )
    assert len(
        monitor.detect_anomalies(
            current_events=three_warnings,
            prior_24h_events=[],
            same_hour_events=[],
            related_updates=[],
            now=NOW,
            config=config,
        )
    ) == 1


def test_existing_fingerprint_spike_requires_multiplier_and_minimum_delta():
    config = monitor.MonitorConfig()
    current = [
        event("2026-05-10 12:00:00,000 - backend.main - ERROR - TimeoutError: provider timeout", i)
        for i in range(6)
    ]
    baseline = [
        event(
            "2026-05-09 12:00:00,000 - backend.main - ERROR - TimeoutError: provider timeout",
            60 + i,
        )
        for i in range(2)
    ]

    anomalies = monitor.detect_anomalies(
        current_events=current,
        prior_24h_events=baseline,
        same_hour_events=[],
        related_updates=[],
        now=NOW,
        config=config,
    )

    assert len(anomalies) == 1
    assert anomalies[0].classification == "spike"
    assert anomalies[0].current_count == 6
    assert anomalies[0].baseline_count == 2


class FakeGitHubClient:
    def __init__(self):
        self.labels = []
        self.created = []
        self.updated = []
        self.comments = []
        self.existing = {}

    def ensure_label(self, name, color, description):
        self.labels.append((name, color, description))

    def find_open_issue_by_fingerprint(self, fingerprint_key):
        return self.existing.get(fingerprint_key)

    def create_issue(self, title, body, labels):
        self.created.append((title, body, labels))
        return {"number": 42, "body": body}

    def update_issue(self, number, title, body):
        self.updated.append((number, title, body))

    def add_comment(self, number, body):
        self.comments.append((number, body))


def test_issue_reporter_creates_labels_and_one_issue_per_new_fingerprint():
    github = FakeGitHubClient()
    anomaly = monitor.Anomaly(
        fingerprint=monitor.Fingerprint(
            key="abc",
            severity="ERROR",
            normalized_message="ERROR RuntimeError failed export",
            exception_type="RuntimeError",
            endpoint=None,
            subsystem="backend.main",
        ),
        classification="new_error",
        confidence="deployment-correlated regression candidate",
        first_seen=NOW,
        last_seen=NOW,
        current_count=1,
        baseline_count=0,
        same_hour_count=0,
        sample_messages=("ERROR RuntimeError failed export",),
        log_sources=("/aws/ec2/cadagent/service",),
        related_updates=(),
    )

    monitor.publish_anomalies(github, [anomaly], now=NOW, config=monitor.MonitorConfig())

    assert ("prod-monitor", "5319e7", "Generated by production error monitor") in github.labels
    assert len(github.created) == 1
    assert "abc" in github.created[0][1]
    assert "deployment-correlated regression candidate" in github.created[0][1]
    assert github.created[0][2] == ["prod-monitor", "prod-anomaly", "automated"]


def test_existing_issue_gets_body_update_and_hourly_comment_cooldown():
    config = monitor.MonitorConfig()
    anomaly = monitor.Anomaly(
        fingerprint=monitor.Fingerprint(
            key="abc",
            severity="ERROR",
            normalized_message="ERROR RuntimeError failed export",
            exception_type="RuntimeError",
            endpoint=None,
            subsystem="backend.main",
        ),
        classification="spike",
        confidence="possible production regression",
        first_seen=NOW - timedelta(hours=2),
        last_seen=NOW,
        current_count=7,
        baseline_count=1,
        same_hour_count=1,
        sample_messages=("ERROR RuntimeError failed export",),
        log_sources=("/aws/ec2/cadagent/service",),
        related_updates=(),
    )

    github = FakeGitHubClient()
    github.existing["abc"] = {
        "number": 7,
        "body": f"<!-- prod-error-monitor:last-comment-at={int((NOW - timedelta(minutes=30)).timestamp())} -->",
    }
    monitor.publish_anomalies(github, [anomaly], now=NOW, config=config)
    assert len(github.updated) == 1
    assert github.comments == []

    github.existing["abc"] = {
        "number": 7,
        "body": f"<!-- prod-error-monitor:last-comment-at={int((NOW - timedelta(hours=2)).timestamp())} -->",
    }
    monitor.publish_anomalies(github, [anomaly], now=NOW, config=config)
    assert len(github.comments) == 1


def test_monitor_failure_opens_deduped_ops_issue():
    github = FakeGitHubClient()

    monitor.publish_monitor_failure(
        github,
        error=RuntimeError("CloudWatch query failed for /aws/ec2/cadagent"),
        now=NOW,
        config=monitor.MonitorConfig(),
    )

    assert len(github.created) == 1
    assert github.created[0][0] == "Production error monitor failure"
    assert "CloudWatch query failed" in github.created[0][1]
    assert "prod-monitor-failure" in github.created[0][2]
