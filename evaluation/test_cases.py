"""Predefined test cases for the IEEE BSU AI Chatbot evaluation suite."""

# ------------------------------------------------------------------
# Single-turn test Q&A pairs
# ------------------------------------------------------------------

SINGLE_TURN_TESTS = [
    {
        "question": "What is IEEE Beni-Suef Student Branch?",
        "expected_keywords": ["IEEE", "Beni-Suef", "student", "branch"],
        "expected_source": "about",
    },
    {
        "question": "What committees does the branch have?",
        "expected_keywords": ["committee", "chapter"],
        "expected_source": "committees",
    },
    {
        "question": "Who is the chair of IEEE BSU?",
        "expected_keywords": ["chair"],
        "expected_source": "leadership",
    },
    {
        "question": "How can I join IEEE as a student?",
        "expected_keywords": ["join", "member", "student"],
        "expected_source": "membership",
    },
    {
        "question": "Tell me about the Computer Society chapter",
        "expected_keywords": ["computer", "society"],
        "expected_source": "computer_society",
    },
    {
        "question": "What is IEEE?",
        "expected_keywords": ["IEEE", "engineering", "technology"],
        "expected_source": "",  # may come from web search
    },
    {
        "question": "What events has the branch organized?",
        "expected_keywords": ["event"],
        "expected_source": "events",
    },
    {
        "question": "What is the Computational Intelligence Society?",
        "expected_keywords": ["computational", "intelligence"],
        "expected_source": "",
    },
]


# ------------------------------------------------------------------
# Multi-turn conversation test suites
# ------------------------------------------------------------------

MULTI_TURN_TESTS = [
    {
        "name": "committee_drill_down",
        "turns": [
            {"message": "What committees does IEEE BSU have?"},
            {"message": "Tell me more about the CS chapter"},
            {"message": "Who leads it?"},
        ],
        "final_expected_keywords": ["computer", "society"],
    },
    {
        "name": "membership_journey",
        "turns": [
            {"message": "How do I join IEEE?"},
            {"message": "What are the benefits of student membership?"},
            {"message": "How much does it cost?"},
        ],
        "final_expected_keywords": ["member"],
    },
    {
        "name": "events_and_activities",
        "turns": [
            {"message": "What events are coming up?"},
            {"message": "Tell me about past events"},
            {"message": "How can I participate?"},
        ],
        "final_expected_keywords": ["event"],
    },
    {
        "name": "general_ieee_then_branch",
        "turns": [
            {"message": "What is IEEE?"},
            {"message": "Tell me about the Egypt Section"},
            {"message": "And what about the Beni-Suef branch specifically?"},
        ],
        "final_expected_keywords": ["Beni-Suef", "branch"],
    },
    {
        "name": "tool_routing_test",
        "turns": [
            {"message": "What is IEEE BSU?"},  # should hit KB
            {"message": "What upcoming events are there?"},  # should hit vTools
            {"message": "What is the latest IEEE conference?"},  # should hit web search
        ],
        "final_expected_keywords": ["conference"],
    },
]
