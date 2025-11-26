"""
Build comprehensive query set for O-RAG evaluation.

This script creates realistic queries based on actual vault content,
with validated chunk-level ground truth.
"""

import json
from pathlib import Path
from typing import Optional


def load_chunks() -> dict:
    """Load filtered chunk index."""
    with open(Path(__file__).parent.parent / "data/filtered_chunks.json") as f:
        return json.load(f)


def find_chunks(chunks: list, pattern: str, field: str = "id") -> list:
    """Find chunks matching a pattern."""
    return [c for c in chunks if pattern.lower() in c[field].lower()]


def get_chunk_id(chunks: list, pattern: str) -> Optional[str]:
    """Get exact chunk ID matching pattern."""
    matches = find_chunks(chunks, pattern)
    if matches:
        return matches[0]["id"]
    return None


def validate_chunk_ids(chunks: list, query: dict) -> bool:
    """Validate all chunk IDs in a query exist."""
    chunk_ids = set(c["id"] for c in chunks)
    for ec in query.get("expected_chunks", []):
        if ec["chunk_id"] not in chunk_ids:
            print(f"  WARNING: Invalid chunk_id: {ec['chunk_id']}")
            return False
    return True


def build_queries(chunks: list) -> list:
    """Build comprehensive query set."""
    queries = []

    # Helper to add query with validation
    def add_query(q: dict):
        if validate_chunk_ids(chunks, q):
            queries.append(q)
        else:
            print(f"  Skipping query {q['id']}")

    # ===========================================
    # PERSON LOOKUPS (simple-search)
    # ===========================================

    add_query({
        "id": "simple-001",
        "tool": "simple-search",
        "query": "Ritu Goel",
        "category": "person-lookup",
        "expected_chunks": [
            {"chunk_id": "People/Ritu Goel.md::root", "relevance": 3,
             "reason": "Person page - direct match"}
        ],
        "notes": "Exact name lookup"
    })

    add_query({
        "id": "simple-002",
        "tool": "simple-search",
        "query": "Brian Eriksson",
        "category": "person-lookup",
        "expected_chunks": [
            {"chunk_id": "People/Brian Eriksson.md::Brain Eriksson", "relevance": 3,
             "reason": "Person page (note: typo in heading)"}
        ],
        "notes": "Exact name - tests handling of typos in headings"
    })

    add_query({
        "id": "simple-003",
        "tool": "simple-search",
        "query": "Kosta Blank",
        "category": "person-lookup",
        "expected_chunks": [
            {"chunk_id": "People/Kosta Blank.md::Kosta Blank", "relevance": 3,
             "reason": "Person page"}
        ],
        "notes": "Exact name lookup"
    })

    add_query({
        "id": "simple-004",
        "tool": "simple-search",
        "query": "Jayant Kumar",
        "category": "person-lookup",
        "expected_chunks": [
            {"chunk_id": "People/Jayant Kumar.md::Jayant Kumar", "relevance": 3,
             "reason": "Person page"}
        ],
        "notes": "Exact name lookup"
    })

    add_query({
        "id": "simple-005",
        "tool": "simple-search",
        "query": "Vipul Dalal",
        "category": "person-lookup",
        "expected_chunks": [
            {"chunk_id": "People/Vipul Dalal.md::root", "relevance": 3,
             "reason": "Person page - VP of SDC"}
        ],
        "notes": "Executive lookup"
    })

    add_query({
        "id": "simple-006",
        "tool": "simple-search",
        "query": "Subhajit",
        "category": "person-lookup",
        "expected_chunks": [
            {"chunk_id": "People/Subhajit.md::Subhajit Sanyal", "relevance": 3,
             "reason": "Person page - first name match"}
        ],
        "notes": "First name only lookup"
    })

    # ===========================================
    # PERSON LOOKUPS (smart-search for roles/context)
    # ===========================================

    add_query({
        "id": "smart-001",
        "tool": "smart-search",
        "query": "who is my manager",
        "category": "person-lookup",
        "expected_chunks": [
            {"chunk_id": "People/Ritu Goel.md::root", "relevance": 3,
             "reason": "Contains 'Ritu is my direct manager'"}
        ],
        "notes": "Semantic query about reporting relationship"
    })

    add_query({
        "id": "smart-002",
        "tool": "smart-search",
        "query": "who leads Query Understanding",
        "category": "person-lookup",
        "expected_chunks": [
            {"chunk_id": "Notes/Programs/Intent/Query Understanding/Query Understanding.md::Query Understanding Home", "relevance": 3,
             "reason": "Contains 'Lead by myself, Subhajit, and Jayant'"},
            {"chunk_id": "People/Subhajit.md::Subhajit Sanyal", "relevance": 2,
             "reason": "QU team member"},
            {"chunk_id": "People/Jayant Kumar.md::Jayant Kumar", "relevance": 2,
             "reason": "QU team member"}
        ],
        "notes": "Leadership query"
    })

    add_query({
        "id": "smart-003",
        "tool": "smart-search",
        "query": "VP of SDC",
        "category": "person-lookup",
        "expected_chunks": [
            {"chunk_id": "People/Vipul Dalal.md::root", "relevance": 3,
             "reason": "Contains 'VP for SDC'"}
        ],
        "notes": "Executive role lookup"
    })

    add_query({
        "id": "smart-004",
        "tool": "smart-search",
        "query": "Brian engineering counterpart recommendations",
        "category": "person-lookup",
        "expected_chunks": [
            {"chunk_id": "People/Brian Eriksson.md::Brain Eriksson", "relevance": 3,
             "reason": "Contains 'Director of Engineering and my direct eng counterpart for the recommendations program'"}
        ],
        "notes": "Role + program context"
    })

    add_query({
        "id": "smart-005",
        "tool": "smart-search",
        "query": "who is leading Firefly semantic search",
        "category": "person-lookup",
        "expected_chunks": [
            {"chunk_id": "Notes/Periodic Notes/Daily Notes/November 20, 2025.md::Meetings>Semantic Search for [[Firefly]] #meetings", "relevance": 3,
             "reason": "Contains 'Rhut is leading this one'"}
        ],
        "notes": "Project leadership query"
    })

    # ===========================================
    # MEETING RECALL
    # ===========================================

    add_query({
        "id": "smart-010",
        "tool": "smart-search",
        "query": "Ritu 1x1",
        "category": "meeting-recall",
        "expected_chunks": [
            {"chunk_id": "Notes/Periodic Notes/Daily Notes/November 19, 2025.md::Meetings>[[Ritu Goel|Ritu]] #meetings/1x1", "relevance": 3,
             "reason": "Recent Ritu 1x1"},
            {"chunk_id": "Notes/Periodic Notes/Daily Notes/November 19, 2025.md::Meetings>[[Ritu Goel|Ritu]] #meetings/1x1>Interns", "relevance": 2,
             "reason": "Subtopic of Ritu 1x1"},
            {"chunk_id": "Notes/Periodic Notes/Daily Notes/November 19, 2025.md::Meetings>[[Ritu Goel|Ritu]] #meetings/1x1>Manager move", "relevance": 2,
             "reason": "Subtopic of Ritu 1x1"},
            {"chunk_id": "Notes/Periodic Notes/Daily Notes/November 19, 2025.md::Meetings>[[Ritu Goel|Ritu]] #meetings/1x1>[[Lr Home|Lr]]", "relevance": 2,
             "reason": "Subtopic of Ritu 1x1"},
            {"chunk_id": "Notes/Periodic Notes/Daily Notes/November 19, 2025.md::Meetings>[[Ritu Goel|Ritu]] #meetings/1x1>[[Query Understanding|QU]]", "relevance": 2,
             "reason": "Subtopic of Ritu 1x1"}
        ],
        "notes": "Person + meeting type - should find recent 1x1s"
    })

    add_query({
        "id": "smart-011",
        "tool": "smart-search",
        "query": "Brian 1x1",
        "category": "meeting-recall",
        "expected_chunks": [
            {"chunk_id": "Notes/Periodic Notes/Daily Notes/November 19, 2025.md::Meetings>[[Brian Eriksson|Brian]] #meetings/1x1>Styles", "relevance": 3,
             "reason": "Brian 1x1 - Styles topic"},
            {"chunk_id": "Notes/Periodic Notes/Daily Notes/November 19, 2025.md::Meetings>[[Brian Eriksson|Brian]] #meetings/1x1>[[Photoshop]]", "relevance": 3,
             "reason": "Brian 1x1 - Photoshop topic"},
            {"chunk_id": "Notes/Periodic Notes/Daily Notes/November 19, 2025.md::Meetings>[[Brian Eriksson|Brian]] #meetings/1x1>Intent", "relevance": 3,
             "reason": "Brian 1x1 - Intent topic"}
        ],
        "notes": "First name + meeting type"
    })

    add_query({
        "id": "smart-012",
        "tool": "smart-search",
        "query": "Kosta meeting",
        "category": "meeting-recall",
        "expected_chunks": [
            {"chunk_id": "Notes/Periodic Notes/Daily Notes/November 20, 2025.md::Meetings>[[Kosta Blank|Kosta]] #meetings/1x1", "relevance": 3,
             "reason": "Recent Kosta 1x1"}
        ],
        "notes": "Person + generic meeting"
    })

    add_query({
        "id": "smart-013",
        "tool": "smart-search",
        "query": "Ritu staff meeting",
        "category": "meeting-recall",
        "expected_chunks": [
            {"chunk_id": "Notes/Periodic Notes/Daily Notes/November 20, 2025.md::Meetings>[[Ritu Goel|Ritu]] Staff #meetings", "relevance": 3,
             "reason": "Nov 20 Ritu staff"},
            {"chunk_id": "Notes/Periodic Notes/Daily Notes/November 13, 2025.md::Meetings>[[Ritu Goel|Ritu]] Staff #meetings>Q1 planning", "relevance": 3,
             "reason": "Nov 13 Ritu staff - Q1 planning"}
        ],
        "notes": "Person + specific meeting type"
    })

    add_query({
        "id": "smart-014",
        "tool": "smart-search",
        "query": "what did I discuss with Asim",
        "category": "meeting-recall",
        "expected_chunks": [
            {"chunk_id": "Notes/Periodic Notes/Daily Notes/November 12, 2025.md::Meetings>[[Asim Kadav|Asim]] #meetings/1x1>[[Style Home|Photo styles]]", "relevance": 3,
             "reason": "Asim 1x1 - Photo styles"},
            {"chunk_id": "Notes/Periodic Notes/Daily Notes/November 12, 2025.md::Meetings>[[Asim Kadav|Asim]] #meetings/1x1>[[Query Understanding|QU]]", "relevance": 3,
             "reason": "Asim 1x1 - QU"},
            {"chunk_id": "Notes/Periodic Notes/Daily Notes/November 12, 2025.md::Meetings>[[Asim Kadav|Asim]] #meetings/1x1>[[Photoshop|Ps Web]] presets [[Recommendations Home|recs]] demo", "relevance": 3,
             "reason": "Asim 1x1 - Ps Web recs demo"}
        ],
        "notes": "Open-ended meeting content query"
    })

    add_query({
        "id": "smart-015",
        "tool": "smart-search",
        "query": "Ayush catchup",
        "category": "meeting-recall",
        "expected_chunks": [
            {"chunk_id": "Notes/Periodic Notes/Daily Notes/November 13, 2025.md::Meetings>[[Ayush Jaiswal|Ayush]] catchup #meetings/1x1", "relevance": 3,
             "reason": "Ayush catchup meeting"}
        ],
        "notes": "Person + informal meeting type"
    })

    # Tag-based meeting search
    add_query({
        "id": "simple-010",
        "tool": "simple-search",
        "query": "#meetings/1x1",
        "category": "meeting-recall",
        "expected_chunks": [
            {"chunk_id": "Notes/Periodic Notes/Daily Notes/November 20, 2025.md::Meetings>[[Kosta Blank|Kosta]] #meetings/1x1", "relevance": 3,
             "reason": "Tagged 1x1"},
            {"chunk_id": "Notes/Periodic Notes/Daily Notes/November 19, 2025.md::Meetings>[[Ritu Goel|Ritu]] #meetings/1x1", "relevance": 3,
             "reason": "Tagged 1x1"},
            {"chunk_id": "Notes/Periodic Notes/Daily Notes/November 13, 2025.md::Meetings>[[Ayush Jaiswal|Ayush]] catchup #meetings/1x1", "relevance": 3,
             "reason": "Tagged 1x1"}
        ],
        "notes": "Tag-based filtering for 1x1s"
    })

    add_query({
        "id": "simple-011",
        "tool": "simple-search",
        "query": "#meetings/interviewer",
        "category": "meeting-recall",
        "expected_chunks": [
            {"chunk_id": "Notes/Periodic Notes/Daily Notes/November 14, 2025.md::Meetings>Armando #meetings/interviewer", "relevance": 3,
             "reason": "Interview meeting"},
            {"chunk_id": "Notes/Periodic Notes/Daily Notes/November 13, 2025.md::Meetings>Maria Mora #meetings/interviewer", "relevance": 3,
             "reason": "Interview meeting"},
            {"chunk_id": "Notes/Periodic Notes/Daily Notes/November 13, 2025.md::Meetings>Sophia Gao #meetings/interviewer", "relevance": 3,
             "reason": "Interview meeting"}
        ],
        "notes": "Tag-based filtering for interviews"
    })

    # ===========================================
    # PROJECT STATUS
    # ===========================================

    add_query({
        "id": "smart-020",
        "tool": "smart-search",
        "query": "Intent AI 2026 strategy",
        "category": "project-status",
        "expected_chunks": [
            {"chunk_id": "Notes/Periodic Notes/Daily Notes/November 20, 2025.md::Meetings>[[Intent AI Home|Intent]] Strategy 2026 White-paper #meetings", "relevance": 3,
             "reason": "Strategy whitepaper meeting with Ayush"},
            {"chunk_id": "Notes/Programs/Intent/Intent AI Home.md::2025", "relevance": 2,
             "reason": "Intent AI hub - current year context"}
        ],
        "notes": "Program + year + strategic planning"
    })

    add_query({
        "id": "smart-021",
        "tool": "smart-search",
        "query": "Q1 2026 must nails",
        "category": "project-status",
        "expected_chunks": [
            {"chunk_id": "Notes/Periodic Notes/Daily Notes/November 18, 2025.md::Notes>Q1 2026 Must Nails>[[Recommendations Home|Recs]]", "relevance": 3,
             "reason": "Q1 must nails - Recs"},
            {"chunk_id": "Notes/Periodic Notes/Daily Notes/November 18, 2025.md::Notes>Q1 2026 Must Nails>[[Lr Home|Lr]]", "relevance": 3,
             "reason": "Q1 must nails - Lr"},
            {"chunk_id": "Notes/Periodic Notes/Daily Notes/November 18, 2025.md::Notes>Q1 2026 Must Nails>Intent / [[Query Understanding|QU]]", "relevance": 3,
             "reason": "Q1 must nails - Intent/QU"}
        ],
        "notes": "Planning priorities - vault-specific terminology"
    })

    add_query({
        "id": "smart-022",
        "tool": "smart-search",
        "query": "Lightroom Desktop rollout",
        "category": "project-status",
        "expected_chunks": [
            {"chunk_id": "Notes/Periodic Notes/Daily Notes/November 12, 2025.md::Meetings>Semantic Search v2 #meetings", "relevance": 3,
             "reason": "Lr Desktop GA April 2026 discussion"},
            {"chunk_id": "Notes/Periodic Notes/Daily Notes/November 17, 2025.md::Meetings>Semantic Search in [[Lr Home|Lr]] Desktop #meetings", "relevance": 3,
             "reason": "Lr Desktop semantic search requirements"},
            {"chunk_id": "Notes/Programs/Lightroom/Lr Home.md::Lightroom Home", "relevance": 2,
             "reason": "Hub page with LrD GA April 2026 link"}
        ],
        "notes": "Product + surface + rollout status"
    })

    add_query({
        "id": "smart-023",
        "tool": "smart-search",
        "query": "Photoshop Web recommendations",
        "category": "project-status",
        "expected_chunks": [
            {"chunk_id": "Notes/Periodic Notes/Daily Notes/November 20, 2025.md::Meetings>[[Kosta Blank|Kosta]] #meetings/1x1", "relevance": 3,
             "reason": "Contains 'putting Ps Web recs higher in my must nails for Q1'"},
            {"chunk_id": "Notes/Periodic Notes/Daily Notes/November 19, 2025.md::Meetings>[[Brian Eriksson|Brian]] #meetings/1x1>[[Photoshop]]", "relevance": 2,
             "reason": "Brian 1x1 - Photoshop discussion about Ps Mobile vs Ps Web"}
        ],
        "notes": "Product + surface + feature"
    })

    add_query({
        "id": "smart-024",
        "tool": "smart-search",
        "query": "DC Recs evaluation strategy",
        "category": "project-status",
        "expected_chunks": [
            {"chunk_id": "Notes/Periodic Notes/Daily Notes/November 20, 2025.md::Meetings>[[Acrobat|DC]] [[Recommendations Home|Recs]] sync #meetings", "relevance": 3,
             "reason": "Contains 'They have an eval strat on paper, pending our sign off'"}
        ],
        "notes": "Product abbreviation + feature + specific ask"
    })

    add_query({
        "id": "smart-025",
        "tool": "smart-search",
        "query": "Express 2026 priorities",
        "category": "project-status",
        "expected_chunks": [
            {"chunk_id": "Notes/Periodic Notes/Daily Notes/November 18, 2025.md::Meetings>[[Express]] must nails for 2026 #meetings>Exec summary:", "relevance": 3,
             "reason": "Express must nails exec summary"},
            {"chunk_id": "Notes/Periodic Notes/Daily Notes/November 18, 2025.md::Meetings>[[Express]] must nails for 2026 #meetings>Must nails:", "relevance": 3,
             "reason": "Express must nails details"}
        ],
        "notes": "Product + year + priorities"
    })

    add_query({
        "id": "smart-026",
        "tool": "smart-search",
        "query": "CKG 4.0 production use case",
        "category": "project-status",
        "expected_chunks": [
            {"chunk_id": "Notes/Periodic Notes/Daily Notes/November 19, 2025.md::Meetings>[[Brian Eriksson|Brian]] #meetings/1x1>Intent", "relevance": 3,
             "reason": "Contains 'Prod use case for CKG 4 ... DC?'"},
            {"chunk_id": "Notes/Programs/Intent/CKG 4.0.md::[[Intent AI Home|CKG]] 4.0>Technical Architecture>Use Cases", "relevance": 2,
             "reason": "CKG 4.0 use cases documentation"}
        ],
        "notes": "Technical project + deployment question"
    })

    add_query({
        "id": "smart-027",
        "tool": "smart-search",
        "query": "photo styles status",
        "category": "project-status",
        "expected_chunks": [
            {"chunk_id": "Notes/Periodic Notes/Daily Notes/November 20, 2025.md::Meetings>[[Kosta Blank|Kosta]] #meetings/1x1", "relevance": 3,
             "reason": "Contains 'Photo styles, where are we at?'"},
            {"chunk_id": "Notes/Periodic Notes/Daily Notes/November 12, 2025.md::Meetings>[[Asim Kadav|Asim]] #meetings/1x1>[[Style Home|Photo styles]]", "relevance": 2,
             "reason": "Contains 'No one is on the same page on style'"},
            {"chunk_id": "Notes/Programs/Recommendations/Style Home.md::Style Description", "relevance": 2,
             "reason": "Style Home - problem statement"}
        ],
        "notes": "Feature status query"
    })

    add_query({
        "id": "smart-028",
        "tool": "smart-search",
        "query": "Universal Asset Browser scope",
        "category": "project-status",
        "expected_chunks": [
            {"chunk_id": "Notes/Periodic Notes/Daily Notes/November 20, 2025.md::Meetings>Semantic Search for [[Firefly]] #meetings", "relevance": 3,
             "reason": "Contains UAB scope details - 'Universal Asset Browser' for CC Home search"}
        ],
        "notes": "Project scope query"
    })

    add_query({
        "id": "smart-029",
        "tool": "smart-search",
        "query": "contextual recs peak load RPS",
        "category": "project-status",
        "expected_chunks": [
            {"chunk_id": "Notes/Periodic Notes/Daily Notes/November 20, 2025.md::Meetings>[[Kosta Blank|Kosta]] #meetings/1x1", "relevance": 3,
             "reason": "Contains '120RPS is the current peak load on recs'"}
        ],
        "notes": "Technical metric query"
    })

    # ===========================================
    # HUB DISCOVERY
    # ===========================================

    add_query({
        "id": "smart-030",
        "tool": "smart-search",
        "query": "Intent AI",
        "category": "hub-discovery",
        "expected_chunks": [
            {"chunk_id": "Notes/Programs/Intent/Intent AI Home.md::2025", "relevance": 3,
             "reason": "Intent AI hub - 2025 section"},
            {"chunk_id": "Notes/Programs/Intent/CKG 4.0.md::[[Intent AI Home|CKG]] 4.0", "relevance": 2,
             "reason": "CKG 4.0 - part of Intent AI"}
        ],
        "notes": "Program hub discovery"
    })

    add_query({
        "id": "smart-031",
        "tool": "smart-search",
        "query": "Lightroom",
        "category": "hub-discovery",
        "expected_chunks": [
            {"chunk_id": "Notes/Programs/Lightroom/Lr Home.md::Lightroom Home", "relevance": 3,
             "reason": "Lightroom hub page"},
            {"chunk_id": "Notes/Programs/Lightroom/Lr Home.md::2025", "relevance": 2,
             "reason": "Lightroom 2025 section"}
        ],
        "notes": "Product hub discovery"
    })

    add_query({
        "id": "smart-032",
        "tool": "smart-search",
        "query": "Recommendations",
        "category": "hub-discovery",
        "expected_chunks": [
            {"chunk_id": "Notes/Programs/Recommendations/Recommendations Home.md::Recommendations Home", "relevance": 3,
             "reason": "Recommendations hub page"}
        ],
        "notes": "Program hub discovery"
    })

    add_query({
        "id": "smart-033",
        "tool": "smart-search",
        "query": "Query Understanding",
        "category": "hub-discovery",
        "expected_chunks": [
            {"chunk_id": "Notes/Programs/Intent/Query Understanding/Query Understanding.md::Query Understanding Home", "relevance": 3,
             "reason": "QU hub page"},
            {"chunk_id": "Notes/Programs/Intent/Query Understanding/Query Understanding.md::Overview", "relevance": 2,
             "reason": "QU overview"}
        ],
        "notes": "Technical area hub discovery"
    })

    add_query({
        "id": "smart-034",
        "tool": "smart-search",
        "query": "NER and SRL",
        "category": "hub-discovery",
        "expected_chunks": [
            {"chunk_id": "Notes/Programs/Intent/Query Understanding/NER & SRL.md::NER & SRL Home", "relevance": 3,
             "reason": "NER & SRL hub"}
        ],
        "notes": "Technical component discovery"
    })

    add_query({
        "id": "smart-035",
        "tool": "smart-search",
        "query": "style understanding",
        "category": "hub-discovery",
        "expected_chunks": [
            {"chunk_id": "Notes/Programs/Recommendations/Style Home.md::Style Description", "relevance": 3,
             "reason": "Style description section"},
            {"chunk_id": "Notes/Programs/Recommendations/Style Home.md::Style Description>Problem statement", "relevance": 2,
             "reason": "Style problem statement"}
        ],
        "notes": "Capability area discovery"
    })

    # ===========================================
    # TECHNICAL DETAILS
    # ===========================================

    add_query({
        "id": "smart-040",
        "tool": "smart-search",
        "query": "how does the RAG plugin work",
        "category": "technical-detail",
        "expected_chunks": [
            {"chunk_id": "Notes/Misc/Code Projects/customRAG/Smart Connections Enhancement - Custom RAG.md::Smart Connections Enhancement", "relevance": 3,
             "reason": "Custom RAG overview"},
            {"chunk_id": "Notes/Misc/Code Projects/customRAG/Smart Connections Enhancement - Custom RAG.md::Smart Connections Enhancement>Architecture Overview>The Problem", "relevance": 2,
             "reason": "RAG architecture - the problem"},
            {"chunk_id": "Notes/Misc/Code Projects/customRAG/Smart Connections Enhancement - Custom RAG.md::Smart Connections Enhancement>Architecture Overview>Recall Set Formation", "relevance": 2,
             "reason": "RAG architecture - recall formation"}
        ],
        "notes": "Technical implementation query"
    })

    add_query({
        "id": "smart-041",
        "tool": "smart-search",
        "query": "evaluation methodology offline",
        "category": "technical-detail",
        "expected_chunks": [
            {"chunk_id": "Notes/Misc/Evaluation.md::Evaluations>Eval Playbook>Offline Methodology (Automated)>Query", "relevance": 3,
             "reason": "Offline eval methodology - Query"},
            {"chunk_id": "Notes/Misc/Evaluation.md::Evaluations>Eval Playbook>Offline Methodology (Automated)>MM", "relevance": 3,
             "reason": "Offline eval methodology - MM"},
            {"chunk_id": "Notes/Misc/Evaluation.md::Evaluations>Eval Playbook", "relevance": 2,
             "reason": "Eval playbook overview"}
        ],
        "notes": "Process/methodology query"
    })

    add_query({
        "id": "smart-042",
        "tool": "smart-search",
        "query": "SRL demo link",
        "category": "technical-detail",
        "expected_chunks": [
            {"chunk_id": "Notes/Periodic Notes/Daily Notes/November 12, 2025.md::Meetings>[[Asim Kadav|Asim]] #meetings/1x1>[[Query Understanding|QU]]", "relevance": 3,
             "reason": "Contains SRL demo link"}
        ],
        "notes": "Specific resource lookup"
    })

    add_query({
        "id": "smart-043",
        "tool": "smart-search",
        "query": "CKG 4.0 implementation timeline",
        "category": "technical-detail",
        "expected_chunks": [
            {"chunk_id": "Notes/Programs/Intent/CKG 4.0.md::[[Intent AI Home|CKG]] 4.0>Implementation Timeline>Milestone 1 (Current)", "relevance": 3,
             "reason": "CKG 4.0 timeline - current milestone"},
            {"chunk_id": "Notes/Programs/Intent/CKG 4.0.md::[[Intent AI Home|CKG]] 4.0>Implementation Timeline>Production Rollout", "relevance": 3,
             "reason": "CKG 4.0 timeline - production rollout"}
        ],
        "notes": "Project timeline query"
    })

    add_query({
        "id": "smart-044",
        "tool": "smart-search",
        "query": "multi-language support Lightroom",
        "category": "technical-detail",
        "expected_chunks": [
            {"chunk_id": "Notes/Programs/Lightroom/Lr Language Support.md::Lightroom Language Support>Context", "relevance": 3,
             "reason": "Lr language support context"},
            {"chunk_id": "Notes/Programs/Lightroom/Lr Language Support.md::Lightroom Language Support>Key Constraint: Query Understanding Dependency", "relevance": 2,
             "reason": "Key constraint on QU"},
            {"chunk_id": "Notes/Periodic Notes/Daily Notes/November 12, 2025.md::Meetings>Semantic Search v2 #meetings", "relevance": 2,
             "reason": "Contains multi-language ask for Lr Desktop"}
        ],
        "notes": "Feature + product context"
    })

    # ===========================================
    # RECENT ACTIVITY
    # ===========================================

    add_query({
        "id": "smart-050",
        "tool": "smart-search",
        "query": "Lightroom recent updates",
        "category": "recent-activity",
        "expected_chunks": [
            {"chunk_id": "Notes/Periodic Notes/Daily Notes/November 17, 2025.md::Meetings>[[Lr Home|Lr]] sync #meetings", "relevance": 3,
             "reason": "Recent Lr sync meeting"},
            {"chunk_id": "Notes/Periodic Notes/Daily Notes/November 17, 2025.md::Meetings>Semantic Search in [[Lr Home|Lr]] Desktop #meetings", "relevance": 3,
             "reason": "Recent Lr Desktop semantic search meeting"},
            {"chunk_id": "Notes/Periodic Notes/Daily Notes/November 13, 2025.md::Meetings>[[Lr Home|Lightroom]] dashboard #meetings", "relevance": 2,
             "reason": "Lr dashboard meeting"},
            {"chunk_id": "Notes/Periodic Notes/Daily Notes/November 11, 2025.md::Meetings>[[Lr Home|Lr]] Agent kick off #meetings", "relevance": 2,
             "reason": "Lr Agent kickoff"}
        ],
        "notes": "Product + recency signal"
    })

    add_query({
        "id": "smart-051",
        "tool": "smart-search",
        "query": "what happened this week with Intent",
        "category": "recent-activity",
        "expected_chunks": [
            {"chunk_id": "Notes/Periodic Notes/Daily Notes/November 20, 2025.md::Meetings>[[Intent AI Home|Intent]] Strategy 2026 White-paper #meetings", "relevance": 3,
             "reason": "Recent Intent strategy meeting"},
            {"chunk_id": "Notes/Periodic Notes/Daily Notes/November 19, 2025.md::Meetings>[[Brian Eriksson|Brian]] #meetings/1x1>Intent", "relevance": 2,
             "reason": "Brian 1x1 Intent discussion"}
        ],
        "notes": "Temporal + program query"
    })

    add_query({
        "id": "smart-052",
        "tool": "smart-search",
        "query": "intern interviews",
        "category": "recent-activity",
        "expected_chunks": [
            {"chunk_id": "Notes/Periodic Notes/Daily Notes/November 14, 2025.md::Meetings>Armando #meetings/interviewer", "relevance": 3,
             "reason": "Armando interview - verdict: yes!"},
            {"chunk_id": "Notes/Periodic Notes/Daily Notes/November 13, 2025.md::Meetings>Maria Mora #meetings/interviewer", "relevance": 3,
             "reason": "Maria interview - verdict: yes"},
            {"chunk_id": "Notes/Periodic Notes/Daily Notes/November 13, 2025.md::Meetings>Sophia Gao #meetings/interviewer", "relevance": 3,
             "reason": "Sophia interview - verdict: yes"},
            {"chunk_id": "Notes/Periodic Notes/Daily Notes/November 14, 2025.md::Meetings>Sean Park #meetings/interviewer", "relevance": 2,
             "reason": "Sean interview - verdict: no"},
            {"chunk_id": "Notes/Programs/2026 Summer Internships - Ritu's Team.md::2026 Summer Internships - Ritu's Team>Candidates", "relevance": 2,
             "reason": "Intern candidates list"}
        ],
        "notes": "Activity type query"
    })

    # ===========================================
    # LINK/RESOURCE LOOKUPS
    # ===========================================

    add_query({
        "id": "smart-060",
        "tool": "smart-search",
        "query": "CPro strategy video link",
        "category": "technical-detail",
        "expected_chunks": [
            {"chunk_id": "Notes/Periodic Notes/Daily Notes/November 20, 2025.md::Meetings>[[Ritu Goel|Ritu]] Staff #meetings", "relevance": 3,
             "reason": "Contains CPro strategy deep dive IATV link"}
        ],
        "notes": "Specific resource lookup"
    })

    add_query({
        "id": "smart-061",
        "tool": "smart-search",
        "query": "Intent AI 2026 proposal doc",
        "category": "technical-detail",
        "expected_chunks": [
            {"chunk_id": "Notes/Periodic Notes/Daily Notes/November 20, 2025.md::Meetings>[[Intent AI Home|Intent]] Strategy 2026 White-paper #meetings", "relevance": 3,
             "reason": "Contains Ayush's proposal doc SharePoint link"}
        ],
        "notes": "Document link lookup"
    })

    add_query({
        "id": "smart-062",
        "tool": "smart-search",
        "query": "Lr cost optimization wiki",
        "category": "technical-detail",
        "expected_chunks": [
            {"chunk_id": "Notes/Periodic Notes/Daily Notes/November 12, 2025.md::Meetings>[[Lr Home|Lr]] cost chat with [[Chhayakanta Padhi|Chhaya]] #meetings/1x1", "relevance": 3,
             "reason": "Contains LR Cost Optimization Guide wiki link"}
        ],
        "notes": "Wiki link lookup"
    })

    add_query({
        "id": "smart-063",
        "tool": "smart-search",
        "query": "Ps Web presets demo video",
        "category": "technical-detail",
        "expected_chunks": [
            {"chunk_id": "Notes/Periodic Notes/Daily Notes/November 12, 2025.md::Meetings>[[Asim Kadav|Asim]] #meetings/1x1>[[Photoshop|Ps Web]] presets [[Recommendations Home|recs]] demo", "relevance": 3,
             "reason": "Contains Rohith's demo video Slack link"}
        ],
        "notes": "Demo video lookup"
    })

    # ===========================================
    # SIMPLE SEARCH - EXACT MATCHES
    # ===========================================

    add_query({
        "id": "simple-020",
        "tool": "simple-search",
        "query": "APS",
        "category": "hub-discovery",
        "expected_chunks": [
            {"chunk_id": "Notes/Misc/APS.md::APS", "relevance": 3,
             "reason": "APS page - if exists"}
        ],
        "notes": "Acronym lookup"
    })

    add_query({
        "id": "simple-021",
        "tool": "simple-search",
        "query": "PDF2Pres",
        "category": "hub-discovery",
        "expected_chunks": [
            {"chunk_id": "Notes/Misc/PDF2Pres.md::PDF2Pres", "relevance": 3,
             "reason": "PDF2Pres page - if exists"}
        ],
        "notes": "Feature name lookup"
    })

    add_query({
        "id": "simple-022",
        "tool": "simple-search",
        "query": "CKG 4.0",
        "category": "hub-discovery",
        "expected_chunks": [
            {"chunk_id": "Notes/Programs/Intent/CKG 4.0.md::[[Intent AI Home|CKG]] 4.0", "relevance": 3,
             "reason": "CKG 4.0 page"}
        ],
        "notes": "Project version lookup"
    })

    # ===========================================
    # COMPLEX QUERIES
    # ===========================================

    add_query({
        "id": "smart-070",
        "tool": "smart-search",
        "query": "who should I talk to about AdobeOne ranking for Lightroom",
        "category": "person-lookup",
        "expected_chunks": [
            {"chunk_id": "Notes/Periodic Notes/Daily Notes/November 19, 2025.md::Meetings>[[Ritu Goel|Ritu]] #meetings/1x1>[[Lr Home|Lr]]", "relevance": 3,
             "reason": "Contains 'Feng bin has a project to fine tune adobe One V2'"}
        ],
        "notes": "Multi-hop reasoning needed - person for specific task"
    })

    add_query({
        "id": "smart-071",
        "tool": "smart-search",
        "query": "what files are included in Universal Asset Browser search",
        "category": "technical-detail",
        "expected_chunks": [
            {"chunk_id": "Notes/Periodic Notes/Daily Notes/November 20, 2025.md::Meetings>Semantic Search for [[Firefly]] #meetings", "relevance": 3,
             "reason": "Contains list: PSD, AI, FF generations, Express projects, inDesign, PDFs"}
        ],
        "notes": "Specific list/detail lookup"
    })

    add_query({
        "id": "smart-072",
        "tool": "smart-search",
        "query": "what are the requirements for Lr Desktop semantic search",
        "category": "project-status",
        "expected_chunks": [
            {"chunk_id": "Notes/Periodic Notes/Daily Notes/November 17, 2025.md::Meetings>Semantic Search in [[Lr Home|Lr]] Desktop #meetings", "relevance": 3,
             "reason": "Contains requirements ranking: embeddings, language support, relevance improvements"}
        ],
        "notes": "Requirements query"
    })

    add_query({
        "id": "smart-073",
        "tool": "smart-search",
        "query": "which intern candidates were approved",
        "category": "recent-activity",
        "expected_chunks": [
            {"chunk_id": "Notes/Periodic Notes/Daily Notes/November 14, 2025.md::Meetings>Armando #meetings/interviewer", "relevance": 3,
             "reason": "verdict: yes!"},
            {"chunk_id": "Notes/Periodic Notes/Daily Notes/November 13, 2025.md::Meetings>Maria Mora #meetings/interviewer", "relevance": 3,
             "reason": "verdict: yes"},
            {"chunk_id": "Notes/Periodic Notes/Daily Notes/November 13, 2025.md::Meetings>Sophia Gao #meetings/interviewer", "relevance": 3,
             "reason": "verdict: yes"}
        ],
        "notes": "Filtering by outcome"
    })

    return queries


def main():
    data = load_chunks()
    chunks = data["chunk_index"]

    print(f"Loaded {len(chunks)} chunks")
    print()

    queries = build_queries(chunks)

    output = {
        "version": "0.2.0",
        "description": "O-RAG evaluation queries - chunk-level ground truth (comprehensive)",
        "stats": {
            "total_queries": len(queries),
            "by_tool": {},
            "by_category": {}
        },
        "queries": queries
    }

    # Compute stats
    from collections import Counter
    tool_counts = Counter(q["tool"] for q in queries)
    category_counts = Counter(q["category"] for q in queries)

    output["stats"]["by_tool"] = dict(tool_counts)
    output["stats"]["by_category"] = dict(category_counts)

    print(f"Generated {len(queries)} queries")
    print(f"By tool: {dict(tool_counts)}")
    print(f"By category: {dict(category_counts)}")

    output_path = Path(__file__).parent.parent / "data/queries.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
