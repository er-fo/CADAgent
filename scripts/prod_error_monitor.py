#!/usr/bin/env python3
"""Detect production log anomalies and open/update GitHub issues."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable, Mapping, Sequence


SEVERITY_RE = re.compile(r"\b(DEBUG|INFO|WARNING|ERROR|CRITICAL)\b")
EXCEPTION_RE = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*(?:Error|Exception|Timeout))\b")
ENDPOINT_RE = re.compile(r"(?<![A-Za-z0-9_])/[A-Za-z0-9_./{}:-]+")
SUBSYSTEM_RE = re.compile(r"\d{4}-\d\d-\d\d[^\n]*? - ([A-Za-z_][A-Za-z0-9_.]*) - (?:DEBUG|INFO|WARNING|ERROR|CRITICAL) - ")
LAST_COMMENT_RE = re.compile(r"prod-error-monitor:last-comment-at=(\d+)")
FINGERPRINT_RE = re.compile(r"prod-error-monitor:fingerprint=([a-f0-9]{16})")


@dataclass(frozen=True)
class MonitorConfig:
    window_minutes: int = 30
    related_update_lookback_hours: int = 36
    warning_new_min_count: int = 3
    error_new_min_count: int = 1
    spike_multiplier: float = 3.0
    spike_min_delta: int = 3
    comment_cooldown_seconds: int = 3600
    sample_limit: int = 3
    region: str = "eu-north-1"
    repository: str = "er-fo/cadagent-backend-legacy"
    log_groups: tuple[str, ...] = ()
    codedeploy_application: str = "cadagent-backend-legacy"
    codedeploy_deployment_group: str = "cadagent-backend-legacy-ec2"
    labels: tuple[str, ...] = ("prod-monitor", "prod-anomaly", "automated")
    failure_labels: tuple[str, ...] = ("prod-monitor", "prod-monitor-failure", "automated")

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "MonitorConfig":
        kwargs: dict[str, Any] = {}
        for field_name in cls.__dataclass_fields__:  # type: ignore[attr-defined]
            if field_name in data:
                kwargs[field_name] = data[field_name]
        for tuple_field in ("log_groups", "labels", "failure_labels"):
            if tuple_field in kwargs and not isinstance(kwargs[tuple_field], tuple):
                kwargs[tuple_field] = tuple(kwargs[tuple_field])
        return cls(**kwargs)


@dataclass(frozen=True)
class Fingerprint:
    key: str
    severity: str
    normalized_message: str
    exception_type: str | None
    endpoint: str | None
    subsystem: str | None


@dataclass(frozen=True)
class ProductionUpdate:
    source: str
    identifier: str
    deployed_at: datetime
    sha: str | None = None
    url: str | None = None
    title: str | None = None


@dataclass(frozen=True)
class Anomaly:
    fingerprint: Fingerprint
    classification: str
    confidence: str
    first_seen: datetime
    last_seen: datetime
    current_count: int
    baseline_count: int
    same_hour_count: int
    sample_messages: tuple[str, ...]
    log_sources: tuple[str, ...]
    related_updates: tuple[ProductionUpdate, ...] = ()


@dataclass
class FingerprintStats:
    fingerprint: Fingerprint
    count: int = 0
    first_seen: datetime | None = None
    last_seen: datetime | None = None
    sample_messages: list[str] = field(default_factory=list)
    log_sources: set[str] = field(default_factory=set)

    def add_event(self, event: Mapping[str, Any], config: MonitorConfig) -> None:
        occurred_at = event_time(event)
        self.count += 1
        self.first_seen = occurred_at if self.first_seen is None else min(self.first_seen, occurred_at)
        self.last_seen = occurred_at if self.last_seen is None else max(self.last_seen, occurred_at)
        if len(self.sample_messages) < config.sample_limit:
            self.sample_messages.append(redact_sensitive_text(str(event.get("message", ""))))
        source = "/".join(
            part
            for part in (str(event.get("logGroupName", "")), str(event.get("logStreamName", "")))
            if part
        )
        if source:
            self.log_sources.add(source)


def redact_sensitive_text(text: str) -> str:
    redacted = re.sub(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b", "[email]", text)
    redacted = re.sub(r"(?i)(authorization:\s*bearer\s+)[A-Za-z0-9._=-]+", r"\1[secret]", redacted)
    redacted = re.sub(r"(?i)\b(api[_-]?key|token|secret|password)=['\"]?[^'\"\s]+", r"\1=[secret]", redacted)
    redacted = re.sub(r"\b(sk-[A-Za-z0-9_-]{8,}|eyJ[A-Za-z0-9._=-]{16,})\b", "[secret]", redacted)
    return redacted


def event_time(event: Mapping[str, Any]) -> datetime:
    value = event.get("timestamp")
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(value / 1000, tz=timezone.utc)
    return datetime.now(timezone.utc)


def classify_severity(message: str) -> str | None:
    match = SEVERITY_RE.search(message)
    if match:
        return match.group(1)
    lower = message.lower()
    if "traceback" in lower or "exception" in lower or "failed" in lower or "service" in lower and "not active" in lower:
        return "ERROR"
    return None


def is_candidate_event(event: Mapping[str, Any]) -> bool:
    message = str(event.get("message", ""))
    severity = classify_severity(message)
    return severity in {"WARNING", "ERROR", "CRITICAL"}


def normalize_message(message: str) -> str:
    normalized = redact_sensitive_text(message)
    normalized = re.sub(r"\d{4}-\d\d-\d\d[ T]\d\d:\d\d:\d\d(?:[,.]\d+)?(?:Z|[+-]\d\d:?\d\d)?", "<timestamp>", normalized)
    normalized = re.sub(r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b", "<uuid>", normalized, flags=re.I)
    normalized = re.sub(r"\b[0-9a-f]{32,40}\b", "<hex>", normalized, flags=re.I)
    normalized = re.sub(r"\b\d+\b", "<num>", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def build_fingerprint(event: Mapping[str, Any]) -> Fingerprint:
    message = str(event.get("message", ""))
    severity = classify_severity(message) or "UNKNOWN"
    normalized = normalize_message(message)
    exception_match = EXCEPTION_RE.search(message)
    endpoint_match = ENDPOINT_RE.search(message)
    subsystem_match = SUBSYSTEM_RE.search(message)
    context = "|".join(
        part
        for part in (
            severity,
            exception_match.group(1) if exception_match else "",
            endpoint_match.group(0) if endpoint_match else "",
            subsystem_match.group(1) if subsystem_match else "",
            normalized,
        )
        if part
    )
    key = hashlib.sha256(context.encode("utf-8")).hexdigest()[:16]
    return Fingerprint(
        key=key,
        severity=severity,
        normalized_message=normalized,
        exception_type=exception_match.group(1) if exception_match else None,
        endpoint=endpoint_match.group(0) if endpoint_match else None,
        subsystem=subsystem_match.group(1) if subsystem_match else None,
    )


def aggregate_events(events: Iterable[Mapping[str, Any]], config: MonitorConfig) -> dict[str, FingerprintStats]:
    grouped: dict[str, FingerprintStats] = {}
    for event in events:
        if not is_candidate_event(event):
            continue
        fingerprint = build_fingerprint(event)
        stats = grouped.setdefault(fingerprint.key, FingerprintStats(fingerprint=fingerprint))
        stats.add_event(event, config)
    return grouped


def detect_anomalies(
    *,
    current_events: Sequence[Mapping[str, Any]],
    prior_24h_events: Sequence[Mapping[str, Any]],
    same_hour_events: Sequence[Mapping[str, Any]],
    related_updates: Sequence[ProductionUpdate],
    now: datetime,
    config: MonitorConfig,
) -> list[Anomaly]:
    current = aggregate_events(current_events, config)
    prior = aggregate_events(prior_24h_events, config)
    same_hour = aggregate_events(same_hour_events, config)
    anomalies: list[Anomaly] = []

    for key, stats in current.items():
        prior_count = prior.get(key).count if key in prior else 0
        same_hour_count = same_hour.get(key).count if key in same_hour else 0
        baseline_count = max(prior_count, same_hour_count)
        is_new = baseline_count == 0
        severity = stats.fingerprint.severity
        classification: str | None = None

        if is_new and severity in {"ERROR", "CRITICAL"} and stats.count >= config.error_new_min_count:
            classification = "new_error"
        elif is_new and severity == "WARNING" and stats.count >= config.warning_new_min_count:
            classification = "new_warning_spike"
        elif (
            not is_new
            and stats.count >= baseline_count * config.spike_multiplier
            and stats.count - baseline_count >= config.spike_min_delta
        ):
            classification = "spike"

        if not classification:
            continue

        first_seen = stats.first_seen or now
        anomaly_updates = filter_related_updates(
            related_updates,
            first_seen=first_seen,
            lookback_hours=config.related_update_lookback_hours,
        )
        confidence = confidence_for(classification, anomaly_updates)
        anomalies.append(
            Anomaly(
                fingerprint=stats.fingerprint,
                classification=classification,
                confidence=confidence,
                first_seen=first_seen,
                last_seen=stats.last_seen or now,
                current_count=stats.count,
                baseline_count=baseline_count,
                same_hour_count=same_hour_count,
                sample_messages=tuple(stats.sample_messages),
                log_sources=tuple(sorted(stats.log_sources)),
                related_updates=tuple(anomaly_updates),
            )
        )

    return sorted(anomalies, key=lambda item: (item.fingerprint.severity != "CRITICAL", item.first_seen))


def filter_related_updates(
    updates: Sequence[ProductionUpdate],
    *,
    first_seen: datetime,
    lookback_hours: int,
) -> list[ProductionUpdate]:
    earliest = first_seen - timedelta(hours=lookback_hours)
    return [
        update
        for update in updates
        if earliest <= update.deployed_at <= first_seen
    ]


def confidence_for(classification: str, related_updates: Sequence[ProductionUpdate]) -> str:
    if related_updates and classification in {"new_error", "spike"}:
        return "deployment-correlated regression candidate"
    if related_updates:
        return "possibly related production update"
    return "production anomaly"


class CloudWatchLogClient:
    def __init__(self, region: str):
        import boto3

        self.client = boto3.client("logs", region_name=region)

    def fetch_events(self, log_groups: Sequence[str], start: datetime, end: datetime) -> list[dict[str, Any]]:
        start_ms = int(start.timestamp() * 1000)
        end_ms = int(end.timestamp() * 1000)
        events: list[dict[str, Any]] = []
        for group in log_groups:
            kwargs: dict[str, Any] = {
                "logGroupName": group,
                "startTime": start_ms,
                "endTime": end_ms,
                "filterPattern": "?ERROR ?CRITICAL ?WARNING ?Traceback ?Exception ?failed ?Failed",
            }
            while True:
                response = self.client.filter_log_events(**kwargs)
                for event in response.get("events", []):
                    event["logGroupName"] = group
                    events.append(event)
                token = response.get("nextToken")
                if not token:
                    break
                kwargs["nextToken"] = token
        return events


class CodeDeployClient:
    def __init__(self, region: str):
        import boto3

        self.client = boto3.client("codedeploy", region_name=region)

    def related_updates(self, config: MonitorConfig, since: datetime, until: datetime) -> list[ProductionUpdate]:
        updates: list[ProductionUpdate] = []
        kwargs: dict[str, Any] = {
            "applicationName": config.codedeploy_application,
            "deploymentGroupName": config.codedeploy_deployment_group,
            "createTimeRange": {"start": since, "end": until},
            "includeOnlyStatuses": ["Succeeded"],
        }
        while True:
            response = self.client.list_deployments(**kwargs)
            for deployment_id in response.get("deployments", []):
                deployment = self.client.get_deployment(deploymentId=deployment_id).get("deploymentInfo", {})
                deployed_at = deployment.get("completeTime") or deployment.get("createTime") or until
                revision = deployment.get("revision", {}).get("s3Location", {})
                key = revision.get("key", "")
                sha_match = re.search(r"deployment-([0-9a-f]{7,40})\.zip", key)
                updates.append(
                    ProductionUpdate(
                        source="codedeploy",
                        identifier=deployment_id,
                        deployed_at=deployed_at if isinstance(deployed_at, datetime) else until,
                        sha=sha_match.group(1) if sha_match else None,
                        title=config.codedeploy_deployment_group,
                    )
                )
            token = response.get("nextToken")
            if not token:
                break
            kwargs["nextToken"] = token
        return sorted(updates, key=lambda item: item.deployed_at, reverse=True)


class GitHubClient:
    def __init__(self, repository: str, token: str):
        self.repository = repository
        self.token = token
        self.base_url = f"https://api.github.com/repos/{repository}"

    def _request(self, method: str, path: str, payload: Mapping[str, Any] | None = None) -> Any:
        data = json.dumps(payload).encode("utf-8") if payload is not None else None
        request = urllib.request.Request(
            f"{self.base_url}{path}",
            data=data,
            method=method,
            headers={
                "Accept": "application/vnd.github+json",
                "Authorization": f"Bearer {self.token}",
                "Content-Type": "application/json",
                "X-GitHub-Api-Version": "2022-11-28",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                body = response.read().decode("utf-8")
                return json.loads(body) if body else {}
        except urllib.error.HTTPError as exc:
            if exc.code == 404 and method == "POST" and path == "/labels":
                return {}
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"GitHub API {method} {path} failed: {exc.code} {detail}") from exc

    def ensure_label(self, name: str, color: str, description: str) -> None:
        try:
            self._request("POST", "/labels", {"name": name, "color": color, "description": description})
        except RuntimeError as exc:
            if "already_exists" not in str(exc).lower():
                raise

    def find_open_issue_by_fingerprint(self, fingerprint_key: str) -> Mapping[str, Any] | None:
        issues = self._request(
            "GET",
            f"/issues?state=open&labels={urllib.parse.quote('prod-monitor')}&per_page=100",
        )
        for issue in issues:
            body = issue.get("body") or ""
            match = FINGERPRINT_RE.search(body)
            if match and match.group(1) == fingerprint_key:
                return issue
        return None

    def create_issue(self, title: str, body: str, labels: Sequence[str]) -> Mapping[str, Any]:
        return self._request("POST", "/issues", {"title": title, "body": body, "labels": list(labels)})

    def update_issue(self, number: int, title: str, body: str) -> None:
        self._request("PATCH", f"/issues/{number}", {"title": title, "body": body})

    def add_comment(self, number: int, body: str) -> None:
        self._request("POST", f"/issues/{number}/comments", {"body": body})

    def related_deploy_runs(self, since: datetime, until: datetime) -> list[ProductionUpdate]:
        created = urllib.parse.quote(f"{since.isoformat()}..{until.isoformat()}")
        response = self._request(
            "GET",
            f"/actions/workflows/deploy.yml/runs?branch=main&status=success&created={created}&per_page=50",
        )
        return production_updates_from_github_runs(response)


def parse_github_time(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def production_updates_from_github_runs(response: Mapping[str, Any]) -> list[ProductionUpdate]:
    updates: list[ProductionUpdate] = []
    for run in response.get("workflow_runs", []):
        run_id = str(run.get("id", ""))
        if not run_id:
            continue
        updated_at = run.get("updated_at") or run.get("created_at")
        if not updated_at:
            continue
        updates.append(
            ProductionUpdate(
                source="github-actions",
                identifier=run_id,
                deployed_at=parse_github_time(str(updated_at)),
                sha=run.get("head_sha"),
                url=run.get("html_url"),
                title=run.get("display_title") or run.get("name"),
            )
        )
    return sorted(updates, key=lambda item: item.deployed_at, reverse=True)


def ensure_monitor_labels(github: Any, config: MonitorConfig) -> None:
    label_defs = {
        "prod-monitor": ("5319e7", "Generated by production error monitor"),
        "prod-anomaly": ("d73a4a", "Unusual production error or log spike"),
        "automated": ("ededed", "Created or updated automatically"),
        "prod-monitor-failure": ("b60205", "Production monitor failed to run"),
    }
    for label in set(config.labels + config.failure_labels):
        color, description = label_defs.get(label, ("ededed", "Production monitor label"))
        github.ensure_label(label, color, description)


def issue_title(anomaly: Anomaly) -> str:
    prefix = "Prod error spike" if anomaly.classification == "spike" else "Prod error anomaly"
    message = anomaly.fingerprint.normalized_message[:90]
    return f"{prefix}: {message}"


def render_anomaly_body(anomaly: Anomaly, now: datetime, last_comment_at: int | None = None) -> str:
    updates = "\n".join(
        f"- {item.source} `{item.identifier}` at {item.deployed_at.isoformat()} "
        f"{f'(`{item.sha}`)' if item.sha else ''} {item.url or ''}".rstrip()
        for item in anomaly.related_updates
    ) or "- None found in the 36-hour lookback."
    samples = "\n".join(f"```text\n{sample[:1200]}\n```" for sample in anomaly.sample_messages)
    sources = "\n".join(f"- `{source}`" for source in anomaly.log_sources) or "- Unknown"
    hidden_comment = f"<!-- prod-error-monitor:last-comment-at={last_comment_at or 0} -->"
    return f"""<!-- prod-error-monitor:fingerprint={anomaly.fingerprint.key} -->
{hidden_comment}

## Summary

- Classification: `{anomaly.classification}`
- Confidence: `{anomaly.confidence}`
- Severity: `{anomaly.fingerprint.severity}`
- Fingerprint: `{anomaly.fingerprint.key}`
- First seen: `{anomaly.first_seen.isoformat()}`
- Last seen: `{anomaly.last_seen.isoformat()}`
- Current window count: `{anomaly.current_count}`
- Baseline count: `{anomaly.baseline_count}`
- Same-hour 7d count: `{anomaly.same_hour_count}`
- Generated at: `{now.isoformat()}`

## Context

- Exception: `{anomaly.fingerprint.exception_type or "unknown"}`
- Endpoint: `{anomaly.fingerprint.endpoint or "unknown"}`
- Subsystem: `{anomaly.fingerprint.subsystem or "unknown"}`
- Normalized message: `{anomaly.fingerprint.normalized_message}`

## Possibly Related Production Updates

{updates}

## Log Sources

{sources}

## Redacted Samples

{samples}

## Next Diagnostics

- Check the listed CloudWatch log source around the first-seen timestamp.
- Compare affected code paths against the related production updates.
- Keep causality language conservative until the failure is reproduced or traced.
"""


def should_comment(existing_body: str, now: datetime, config: MonitorConfig) -> bool:
    match = LAST_COMMENT_RE.search(existing_body or "")
    if not match:
        return True
    last_comment_at = int(match.group(1))
    return int(now.timestamp()) - last_comment_at >= config.comment_cooldown_seconds


def last_comment_marker(existing_body: str, now: datetime, config: MonitorConfig) -> int:
    if should_comment(existing_body, now, config):
        return int(now.timestamp())
    match = LAST_COMMENT_RE.search(existing_body or "")
    return int(match.group(1)) if match else 0


def publish_anomalies(github: Any, anomalies: Sequence[Anomaly], now: datetime, config: MonitorConfig) -> None:
    ensure_monitor_labels(github, config)
    for anomaly in anomalies:
        existing = github.find_open_issue_by_fingerprint(anomaly.fingerprint.key)
        title = issue_title(anomaly)
        if not existing:
            body = render_anomaly_body(anomaly, now, last_comment_at=int(now.timestamp()))
            github.create_issue(title, body, list(config.labels))
            continue

        number = int(existing["number"])
        old_body = str(existing.get("body") or "")
        add_comment = should_comment(old_body, now, config)
        body = render_anomaly_body(anomaly, now, last_comment_at=last_comment_marker(old_body, now, config))
        github.update_issue(number, title, body)
        if add_comment:
            github.add_comment(
                number,
                f"Recurring anomaly `{anomaly.fingerprint.key}` seen at {now.isoformat()} "
                f"with `{anomaly.current_count}` events in the current window.",
            )


def publish_monitor_failure(github: Any, error: Exception, now: datetime, config: MonitorConfig) -> None:
    ensure_monitor_labels(github, config)
    fingerprint = hashlib.sha256(str(error).encode("utf-8")).hexdigest()[:16]
    title = "Production error monitor failure"
    body = f"""<!-- prod-error-monitor:fingerprint={fingerprint} -->
<!-- prod-error-monitor:last-comment-at={int(now.timestamp())} -->

## Summary

The production error monitor failed to run.

```text
{redact_sensitive_text(str(error))[:2000]}
```

Generated at: `{now.isoformat()}`
"""
    existing = github.find_open_issue_by_fingerprint(fingerprint)
    if existing:
        github.update_issue(int(existing["number"]), title, body)
    else:
        github.create_issue(title, body, list(config.failure_labels))


def load_config(path: str | None) -> MonitorConfig:
    data: dict[str, Any] = {}
    if path:
        with open(path, "r", encoding="utf-8") as handle:
            data.update(json.load(handle))
    env_log_groups = os.getenv("CLOUDWATCH_LOG_GROUPS", "")
    if env_log_groups:
        data["log_groups"] = [item.strip() for item in env_log_groups.split(",") if item.strip()]
    if os.getenv("AWS_REGION"):
        data["region"] = os.environ["AWS_REGION"]
    if os.getenv("GITHUB_REPOSITORY"):
        data["repository"] = os.environ["GITHUB_REPOSITORY"]
    return MonitorConfig.from_mapping(data)


def same_hour_windows(now: datetime, config: MonitorConfig) -> list[tuple[datetime, datetime]]:
    windows: list[tuple[datetime, datetime]] = []
    for days in range(1, 8):
        end = now - timedelta(days=days)
        start = end - timedelta(minutes=config.window_minutes)
        windows.append((start, end))
    return windows


def run_monitor(config: MonitorConfig, *, dry_run: bool = False) -> list[Anomaly]:
    if not config.log_groups:
        raise RuntimeError("No CloudWatch log groups configured. Set CLOUDWATCH_LOG_GROUPS or config log_groups.")
    now = datetime.now(timezone.utc)
    logs = CloudWatchLogClient(config.region)
    deploys = CodeDeployClient(config.region)
    window_start = now - timedelta(minutes=config.window_minutes)
    prior_start = window_start - timedelta(hours=24)
    related_start = now - timedelta(hours=config.related_update_lookback_hours)
    current_events = logs.fetch_events(config.log_groups, window_start, now)
    prior_events = logs.fetch_events(config.log_groups, prior_start, window_start)
    same_hour_events: list[dict[str, Any]] = []
    for start, end in same_hour_windows(now, config):
        same_hour_events.extend(logs.fetch_events(config.log_groups, start, end))
    related_updates = deploys.related_updates(config, related_start, now)
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        related_updates.extend(GitHubClient(config.repository, token).related_deploy_runs(related_start, now))
    anomalies = detect_anomalies(
        current_events=current_events,
        prior_24h_events=prior_events,
        same_hour_events=same_hour_events,
        related_updates=related_updates,
        now=now,
        config=config,
    )
    if not dry_run and anomalies:
        if not token:
            raise RuntimeError("GITHUB_TOKEN is required to publish anomaly issues.")
        publish_anomalies(GitHubClient(config.repository, token), anomalies, now, config)
    return anomalies


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="infra/monitoring/prod-error-monitor.json")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--window-minutes", type=int)
    parser.add_argument("--lookback-hours", type=int)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    config = load_config(args.config)
    if args.window_minutes:
        config = MonitorConfig.from_mapping({**config.__dict__, "window_minutes": args.window_minutes})
    if args.lookback_hours:
        config = MonitorConfig.from_mapping({**config.__dict__, "related_update_lookback_hours": args.lookback_hours})
    now = datetime.now(timezone.utc)
    try:
        anomalies = run_monitor(config, dry_run=args.dry_run)
    except Exception as exc:
        token = os.environ.get("GITHUB_TOKEN")
        if token and not args.dry_run:
            publish_monitor_failure(GitHubClient(config.repository, token), exc, now, config)
        raise
    print(json.dumps({"anomalies": len(anomalies), "dry_run": args.dry_run}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
