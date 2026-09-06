"""RAG and LangSmith evaluation entry points."""

import logging
from typing import Any

from langsmith import Client as LangSmithClient

from evaluation.test_cases import SINGLE_TURN_TESTS, MULTI_TURN_TESTS

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Metrics helpers
# ------------------------------------------------------------------

def _score_relevance(question: str, answer: str, context: str) -> float:
    """Score how relevant the answer is to the question given the context.

    Returns a float between 0 and 1.  Uses simple keyword overlap
    as a lightweight heuristic; swap for an LLM-based grader in
    production.
    """
    q_words = set(question.lower().split())
    a_words = set(answer.lower().split())
    c_words = set(context.lower().split())

    # Fraction of question keywords that appear in the answer
    if not q_words:
        return 0.0
    relevance = len(q_words & a_words) / len(q_words)
    return round(min(relevance * 1.5, 1.0), 4)


def _score_faithfulness(answer: str, context: str) -> float:
    """Score whether the answer stays faithful to the retrieved context.

    Returns a float between 0 and 1.  Simple keyword overlap heuristic.
    """
    a_words = set(answer.lower().split())
    c_words = set(context.lower().split())

    if not a_words:
        return 0.0
    faithfulness = len(a_words & c_words) / len(a_words)
    return round(min(faithfulness * 2.0, 1.0), 4)


def _score_context_precision(results: list[dict], expected_source: str) -> float:
    """Score whether the right source appeared in the top results."""
    if not results or not expected_source:
        return 0.0
    for i, r in enumerate(results):
        if expected_source.lower() in r.get("source", "").lower():
            return round(1.0 / (i + 1), 4)  # reciprocal rank
    return 0.0


# ------------------------------------------------------------------
# Single-turn evaluation
# ------------------------------------------------------------------

def evaluate_single_turn(agent, retriever) -> list[dict[str, Any]]:
    """Run single-turn Q&A tests and return per-question metrics."""
    results = []

    for test in SINGLE_TURN_TESTS:
        question = test["question"]

        # Retrieve context
        docs = retriever.retrieve(question)
        context = "\n".join(d.get("text", "") for d in docs)

        # Get agent answer
        try:
            response = agent.invoke([("human", question)])
            if isinstance(response, dict) and "messages" in response:
                answer = response["messages"][-1].content
            else:
                answer = str(response)
        except Exception as e:
            answer = f"ERROR: {e}"

        # Score
        metrics = {
            "question": question,
            "answer": answer[:200],
            "expected_keywords": test.get("expected_keywords", []),
            "relevance": _score_relevance(question, answer, context),
            "faithfulness": _score_faithfulness(answer, context),
            "context_precision": _score_context_precision(
                docs, test.get("expected_source", "")
            ),
            "num_docs_retrieved": len(docs),
        }

        # Check keyword presence
        answer_lower = answer.lower()
        matched = [kw for kw in test.get("expected_keywords", []) if kw.lower() in answer_lower]
        metrics["keyword_hit_rate"] = (
            round(len(matched) / len(test["expected_keywords"]), 4)
            if test.get("expected_keywords")
            else 1.0
        )

        results.append(metrics)
        logger.info("Evaluated: %s → relevance=%.2f", question[:50], metrics["relevance"])

    return results


# ------------------------------------------------------------------
# Multi-turn evaluation
# ------------------------------------------------------------------

def evaluate_multi_turn(agent) -> list[dict[str, Any]]:
    """Run multi-turn conversation tests and return per-conversation metrics."""
    results = []

    for conv_test in MULTI_TURN_TESTS:
        conversation_name = conv_test["name"]
        messages_so_far = []
        answers = []
        errors = 0

        for turn in conv_test["turns"]:
            messages_so_far.append(("human", turn["message"]))

            try:
                response = agent.invoke(
                    messages_so_far,
                    config={"configurable": {"thread_id": f"eval-{conversation_name}"}},
                )
                if isinstance(response, dict) and "messages" in response:
                    answer = response["messages"][-1].content
                else:
                    answer = str(response)
                answers.append(answer)
                messages_so_far.append(("assistant", answer))
            except Exception as e:
                answers.append(f"ERROR: {e}")
                errors += 1

        # Check final answer against expected keywords
        final_answer = answers[-1] if answers else ""
        expected = conv_test.get("final_expected_keywords", [])
        matched = [kw for kw in expected if kw.lower() in final_answer.lower()]

        results.append({
            "conversation": conversation_name,
            "num_turns": len(conv_test["turns"]),
            "errors": errors,
            "final_keyword_hit_rate": (
                round(len(matched) / len(expected), 4) if expected else 1.0
            ),
            "final_answer_preview": final_answer[:200],
        })
        logger.info("Evaluated conversation: %s → errors=%d", conversation_name, errors)

    return results


# ------------------------------------------------------------------
# Aggregate evaluation
# ------------------------------------------------------------------

def evaluate(agent=None, retriever=None) -> dict[str, Any]:
    """Run the full evaluation suite and return aggregate metrics."""
    output: dict[str, Any] = {}

    if agent and retriever:
        single = evaluate_single_turn(agent, retriever)
        output["single_turn"] = {
            "num_tests": len(single),
            "avg_relevance": round(
                sum(r["relevance"] for r in single) / max(len(single), 1), 4
            ),
            "avg_faithfulness": round(
                sum(r["faithfulness"] for r in single) / max(len(single), 1), 4
            ),
            "avg_keyword_hit_rate": round(
                sum(r["keyword_hit_rate"] for r in single) / max(len(single), 1), 4
            ),
            "details": single,
        }

    if agent:
        multi = evaluate_multi_turn(agent)
        output["multi_turn"] = {
            "num_conversations": len(multi),
            "total_errors": sum(c["errors"] for c in multi),
            "avg_final_keyword_hit_rate": round(
                sum(c["final_keyword_hit_rate"] for c in multi) / max(len(multi), 1),
                4,
            ),
            "details": multi,
        }

    return output
