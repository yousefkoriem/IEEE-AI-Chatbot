from __future__ import annotations

import logging
from datetime import datetime, timedelta
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

def get_recent_runs(settings: Settings, limit: int = 15) -> list[dict]:
    """Fetches recent runs from LangSmith."""
    client = _get_client(settings)
    if not client:
        return []

    try:
        # Fetch root runs (chains) in the project
        runs = list(client.list_runs(
            project_name=settings.langsmith_project,
            execution_order=1, # Root runs only
            limit=limit,
        ))
        
        results = []
        for run in runs:
            # Try to get the question/input
            question = "Unknown"
            if run.inputs:
                question = str(run.inputs.get("question", run.inputs))
                if len(question) > 50:
                    question = question[:47] + "..."
            
            # Latency
            latency_ms = 0
            if run.start_time and run.end_time:
                latency_ms = int((run.end_time - run.start_time).total_seconds() * 1000)
                
            # Feedback
            feedback_score = "None"
            if run.feedback_stats and "user_score" in run.feedback_stats:
                # Feedback stats might contain average or specific scores
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
                "status": run.status
            })
            
        return results
    except Exception as e:
        logger.warning("Error fetching recent runs: %s", e)
        return []

def get_feedback_summary(settings: Settings) -> dict:
    """Aggregates feedback scores (thumbs up/down counts) from recent runs."""
    client = _get_client(settings)
    if not client:
        return {"up": 0, "down": 0, "total": 0}

    try:
        # Get last 100 feedback entries in project
        feedbacks = list(client.list_feedback(
            project_ids=[client.read_project(project_name=settings.langsmith_project).id],
            limit=100
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

def get_latency_stats(settings: Settings) -> dict:
    """Calculates average response times from recent runs."""
    client = _get_client(settings)
    if not client:
        return {"avg_ms": 0}

    try:
        # Get runs from the last 7 days
        start_time = datetime.utcnow() - timedelta(days=7)
        runs = list(client.list_runs(
            project_name=settings.langsmith_project,
            start_time=start_time,
            execution_order=1,
            limit=50
        ))
        
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
