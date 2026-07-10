from __future__ import annotations

import csv
import io
import logging
from collections import Counter
from datetime import datetime, timedelta
from typing import Any

from langsmith import Client as LangSmithClient

from .config import Settings

logger = logging.getLogger(__name__)


def _get_client(settings: Settings) -> LangSmithClient | None:
    if not settings.langsmith_tracing:
        return None
    try:
        return LangSmithClient()
    except Exception as e:
        logger.warning("Failed to initialize LangSmith client: %s", e)
        return None


def _parse_time_range(time_range: str) -> datetime | None:
    now = datetime.utcnow()
    if time_range == "24h":
        return now - timedelta(hours=24)
    elif time_range == "7d":
        return now - timedelta(days=7)
    elif time_range == "30d":
        return now - timedelta(days=30)
    return None


def get_recent_runs(settings: Settings, limit: int = 15, time_range: str = "7d") -> list[dict]:
    """Fetches recent runs from LangSmith."""
    client = _get_client(settings)
    if not client:
        return []

    try:
        start_time = _parse_time_range(time_range)
        kwargs: dict[str, Any] = dict(
            project_name=settings.langsmith_project,
            execution_order=1,
            limit=limit,
        )
        if start_time:
            kwargs["start_time"] = start_time

        runs = list(client.list_runs(**kwargs))

        results = []
        for run in runs:
            question = "Unknown"
            if run.inputs:
                raw = run.inputs.get("question", str(run.inputs))
                question = str(raw)[:50]

            latency_ms = 0
            if run.start_time and run.end_time:
                latency_ms = int((run.end_time - run.start_time).total_seconds() * 1000)

            feedback_score = "None"
            if run.feedback_stats and "user_score" in run.feedback_stats:
                score_stats = run.feedback_stats.get("user_score", {})
                if "avg" in score_stats:
                    val = score_stats["avg"]
                    feedback_score = "👍" if val >= 0.5 else "👎"

            results.append({
                "id": str(run.id),
                "name": run.name,
                "time": run.start_time.strftime("%Y-%m-%d %H:%M:%S") if run.start_time else "Unknown",
                "question": question,
                "latency_ms": latency_ms,
                "feedback": feedback_score,
                "status": run.status,
            })

        return results
    except Exception as e:
        logger.warning("Error fetching recent runs: %s", e)
        return []


def get_feedback_summary(settings: Settings, time_range: str = "7d") -> dict:
    """Aggregates feedback scores (thumbs up/down counts)."""
    client = _get_client(settings)
    if not client:
        return {"up": 0, "down": 0, "total": 0}

    try:
        feedbacks = list(client.list_feedback(
            project_ids=[client.read_project(project_name=settings.langsmith_project).id],
            limit=100,
        ))

        up = 0
        down = 0
        for f in feedbacks:
            if f.key == "user_score" and f.score is not None:
                if f.score >= 0.5:
                    up += 1
                else:
                    down += 1

        return {"up": up, "down": down, "total": up + down}
    except Exception as e:
        logger.warning("Error fetching feedback summary: %s", e)
        return {"up": 0, "down": 0, "total": 0}


def get_latency_stats(settings: Settings, time_range: str = "7d") -> dict:
    """Calculates average response times."""
    client = _get_client(settings)
    if not client:
        return {"avg_ms": 0}

    try:
        start_time = _parse_time_range(time_range)
        kwargs: dict[str, Any] = dict(
            project_name=settings.langsmith_project,
            execution_order=1,
            limit=50,
        )
        if start_time:
            kwargs["start_time"] = start_time

        runs = list(client.list_runs(**kwargs))

        latencies = []
        for run in runs:
            if run.start_time and run.end_time:
                latencies.append((run.end_time - run.start_time).total_seconds() * 1000)

        if not latencies:
            return {"avg_ms": 0}

        avg_ms = sum(latencies) / len(latencies)
        return {"avg_ms": int(avg_ms)}
    except Exception as e:
        logger.warning("Error fetching latency stats: %s", e)
        return {"avg_ms": 0}


def get_top_queries(settings: Settings, limit: int = 10, time_range: str = "7d") -> list[dict]:
    """Returns most frequent user queries."""
    client = _get_client(settings)
    if not client:
        return []

    try:
        start_time = _parse_time_range(time_range)
        kwargs: dict[str, Any] = dict(
            project_name=settings.langsmith_project,
            execution_order=1,
            limit=100,
        )
        if start_time:
            kwargs["start_time"] = start_time

        runs = list(client.list_runs(**kwargs))
        queries: list[str] = []
        latencies_by_query: dict[str, list[int]] = {}

        for run in runs:
            if run.inputs:
                question = str(run.inputs.get("question", ""))
                if question and len(question) > 5:
                    queries.append(question)
                    if run.start_time and run.end_time:
                        ms = int((run.end_time - run.start_time).total_seconds() * 1000)
                        latencies_by_query.setdefault(question, []).append(ms)

        counter = Counter(queries)
        results = []
        for question, count in counter.most_common(limit):
            avg_latency = 0
            if question in latencies_by_query:
                vals = latencies_by_query[question]
                avg_latency = sum(vals) // len(vals)
            results.append({
                "question": question[:60],
                "count": count,
                "avg_latency_ms": avg_latency,
            })
        return results
    except Exception as e:
        logger.warning("Error fetching top queries: %s", e)
        return []


def get_latency_timeseries(settings: Settings, time_range: str = "7d") -> list[dict]:
    """Returns average latency per day."""
    client = _get_client(settings)
    if not client:
        return []

    try:
        start_time = _parse_time_range(time_range)
        kwargs: dict[str, Any] = dict(
            project_name=settings.langsmith_project,
            execution_order=1,
            limit=200,
        )
        if start_time:
            kwargs["start_time"] = start_time

        runs = list(client.list_runs(**kwargs))
        daily: dict[str, list[int]] = {}

        for run in runs:
            if run.start_time and run.end_time:
                day = run.start_time.strftime("%Y-%m-%d")
                ms = int((run.end_time - run.start_time).total_seconds() * 1000)
                daily.setdefault(day, []).append(ms)

        results = []
        for day in sorted(daily.keys()):
            vals = daily[day]
            results.append({
                "date": day,
                "avg_ms": sum(vals) // len(vals),
                "count": len(vals),
            })
        return results
    except Exception as e:
        logger.warning("Error fetching latency timeseries: %s", e)
        return []


def get_feedback_timeseries(settings: Settings, time_range: str = "7d") -> list[dict]:
    """Returns feedback counts per day."""
    client = _get_client(settings)
    if not client:
        return []

    try:
        start_time = _parse_time_range(time_range)
        kwargs: dict[str, Any] = dict(
            project_name=settings.langsmith_project,
            execution_order=1,
            limit=200,
        )
        if start_time:
            kwargs["start_time"] = start_time

        runs = list(client.list_runs(**kwargs))
        daily: dict[str, dict[str, int]] = {}

        for run in runs:
            if run.start_time:
                day = run.start_time.strftime("%Y-%m-%d")
                daily.setdefault(day, {"date": day, "up": 0, "down": 0, "total": 0})
                fb = "None"
                if run.feedback_stats and "user_score" in run.feedback_stats:
                    score_stats = run.feedback_stats.get("user_score", {})
                    if "avg" in score_stats:
                        fb = "up" if score_stats["avg"] >= 0.5 else "down"
                if fb in ("up", "down"):
                    daily[day][fb] += 1
                    daily[day]["total"] += 1

        return [daily[d] for d in sorted(daily.keys())]
    except Exception as e:
        logger.warning("Error fetching feedback timeseries: %s", e)
        return []


def export_runs_csv(settings: Settings, time_range: str = "7d", limit: int = 200) -> str:
    """Exports recent runs as CSV string."""
    runs = get_recent_runs(settings, limit=limit, time_range=time_range)
    if not runs:
        return "No data available."

    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=["time", "question", "latency_ms", "feedback", "status"])
    writer.writeheader()
    for r in runs:
        writer.writerow({
            "time": r["time"],
            "question": r["question"],
            "latency_ms": r["latency_ms"],
            "feedback": r["feedback"],
            "status": r["status"],
        })
    return output.getvalue()
