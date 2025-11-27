#!/usr/bin/env python3
"""Validate query chunk IDs against the actual chunk index."""

import json
from pathlib import Path
from collections import defaultdict


def validate_queries():
    """Validate all chunk IDs in queries_v3.json exist in filtered_chunks.json."""

    data_dir = Path(__file__).parent.parent / "data"

    # Load filtered chunks
    with open(data_dir / "filtered_chunks.json", "r") as f:
        chunks_data = json.load(f)

    # Build set of valid chunk IDs
    valid_ids = set()
    for chunk in chunks_data["chunk_index"]:
        valid_ids.add(chunk["id"])

    print(f"Loaded {len(valid_ids)} valid chunk IDs from filtered_chunks.json")

    # Load queries
    with open(data_dir / "queries_v3.json", "r") as f:
        queries_data = json.load(f)

    queries = queries_data["queries"]
    print(f"Validating {len(queries)} queries...\n")

    # Track issues
    invalid_chunks = []
    stats = defaultdict(lambda: {"count": 0, "chunks": 0})

    for query in queries:
        tool = query["tool"]
        category = query.get("category", "unknown")
        stats[tool]["count"] += 1
        stats[f"category:{category}"]["count"] += 1

        for expected in query.get("expected_chunks", []):
            chunk_id = expected["chunk_id"]
            stats[tool]["chunks"] += 1

            if chunk_id not in valid_ids:
                invalid_chunks.append({
                    "query_id": query["id"],
                    "chunk_id": chunk_id,
                    "query": query["query"]
                })

    # Report
    print("=== STATS ===")
    print(f"Total queries: {len(queries)}")
    print(f"Total expected chunks: {sum(s['chunks'] for k,s in stats.items() if not k.startswith('category:'))}")
    print()

    print("By tool:")
    for tool in ["simple-search", "smart-search", "local-graph"]:
        s = stats[tool]
        print(f"  {tool}: {s['count']} queries, {s['chunks']} expected chunks")
    print()

    print("By category:")
    for key in sorted(stats.keys()):
        if key.startswith("category:"):
            cat = key.replace("category:", "")
            print(f"  {cat}: {stats[key]['count']} queries")
    print()

    if invalid_chunks:
        print("=== INVALID CHUNK IDS ===")
        for item in invalid_chunks:
            print(f"\n  Query: {item['query_id']} ({item['query']})")
            print(f"  Invalid: {item['chunk_id']}")

            # Try to find similar IDs
            chunk_file = item['chunk_id'].split("::")[0] if "::" in item['chunk_id'] else None
            if chunk_file:
                similar = [cid for cid in valid_ids if chunk_file in cid]
                if similar:
                    print(f"  Similar IDs in {chunk_file}:")
                    for s in sorted(similar)[:5]:
                        print(f"    - {s}")

        print(f"\n\nFOUND {len(invalid_chunks)} INVALID CHUNK IDS")
        return False
    else:
        print("=== ALL CHUNK IDS VALID ===")
        return True


if __name__ == "__main__":
    success = validate_queries()
    exit(0 if success else 1)
