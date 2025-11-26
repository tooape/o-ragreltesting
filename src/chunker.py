"""
Hierarchical Markdown Chunker for O-RAG

Chunks markdown files by heading hierarchy (H1-H6), preserving:
- Full heading path for context (e.g., "CKG 4.0 > Overview > Key Features")
- Frontmatter metadata (tags, aliases, related pages)
- Note path for retrieval

Output format matches O-RAG spec:
{
  "title": "CKG 4.0 > Overview",
  "content": "...",
  "notePath": "Notes/Programs/Intent/CKG 4.0.md",
  "headingPath": ["Overview"],
  "metadata": { "tags": [...], "aliases": [...] }
}
"""

import re
import yaml
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional, Any
from datetime import date, datetime
import json


class DateTimeEncoder(json.JSONEncoder):
    """JSON encoder that handles date/datetime objects."""
    def default(self, obj: Any) -> Any:
        if isinstance(obj, (date, datetime)):
            return obj.isoformat()
        return super().default(obj)


@dataclass
class Chunk:
    """A single chunk from a markdown document."""
    title: str  # Hierarchical title: "Note Title > H1 > H2"
    content: str  # The actual text content
    note_path: str  # Relative path to the note
    heading_path: list[str]  # List of headings leading to this chunk
    metadata: dict = field(default_factory=dict)  # Frontmatter metadata

    # For evaluation
    start_line: int = 0
    end_line: int = 0

    def to_dict(self) -> dict:
        return {
            "title": self.title,
            "content": self.content,
            "notePath": self.note_path,
            "headingPath": self.heading_path,
            "metadata": self.metadata,
            "startLine": self.start_line,
            "endLine": self.end_line,
        }

    def token_estimate(self) -> int:
        """Rough token estimate (~4 chars per token)."""
        return len(self.content) // 4


@dataclass
class ChunkedDocument:
    """A fully chunked markdown document."""
    note_path: str
    note_title: str
    chunks: list[Chunk]
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "notePath": self.note_path,
            "noteTitle": self.note_title,
            "metadata": self.metadata,
            "chunks": [c.to_dict() for c in self.chunks],
        }


class MarkdownChunker:
    """
    Chunks markdown files by heading hierarchy.

    Strategy:
    - Parse frontmatter for metadata
    - Split on headings (H1-H6)
    - Build hierarchical title from heading path
    - Minimum chunk size to avoid noise
    """

    HEADING_PATTERN = re.compile(r'^(#{1,6})\s+(.+)$', re.MULTILINE)
    FRONTMATTER_PATTERN = re.compile(r'^---\s*\n(.*?)\n---\s*\n', re.DOTALL)

    def __init__(
        self,
        min_chunk_chars: int = 50,
        include_frontmatter_chunk: bool = False,
    ):
        self.min_chunk_chars = min_chunk_chars
        self.include_frontmatter_chunk = include_frontmatter_chunk

    def chunk_file(self, file_path: Path, base_path: Optional[Path] = None) -> ChunkedDocument:
        """Chunk a single markdown file."""
        content = file_path.read_text(encoding='utf-8')

        # Calculate relative path
        if base_path:
            note_path = str(file_path.relative_to(base_path))
        else:
            note_path = str(file_path)

        # Extract note title from filename
        note_title = file_path.stem

        return self.chunk_content(content, note_path, note_title)

    def chunk_content(self, content: str, note_path: str, note_title: str) -> ChunkedDocument:
        """Chunk markdown content string."""

        # Extract frontmatter
        metadata = {}
        body = content
        fm_match = self.FRONTMATTER_PATTERN.match(content)
        if fm_match:
            try:
                metadata = yaml.safe_load(fm_match.group(1)) or {}
            except yaml.YAMLError:
                metadata = {}
            body = content[fm_match.end():]

        # Normalize metadata keys we care about
        normalized_metadata = {
            "tags": self._extract_tags(metadata),
            "aliases": metadata.get("aliases", []),
            "pageType": metadata.get("pageType", ""),
            "created": metadata.get("created", ""),
            "relatedPages": metadata.get("Related Pages", []),
        }

        # Find all headings with positions
        lines = body.split('\n')
        headings = []  # (line_num, level, text)

        for i, line in enumerate(lines):
            match = self.HEADING_PATTERN.match(line)
            if match:
                level = len(match.group(1))
                text = match.group(2).strip()
                headings.append((i, level, text))

        chunks = []

        # If no headings, entire body is one chunk
        if not headings:
            chunk_content = body.strip()
            if len(chunk_content) >= self.min_chunk_chars:
                chunks.append(Chunk(
                    title=note_title,
                    content=chunk_content,
                    note_path=note_path,
                    heading_path=[],
                    metadata=normalized_metadata,
                    start_line=1,
                    end_line=len(lines),
                ))
        else:
            # Content before first heading
            first_heading_line = headings[0][0]
            if first_heading_line > 0:
                pre_content = '\n'.join(lines[:first_heading_line]).strip()
                if len(pre_content) >= self.min_chunk_chars:
                    chunks.append(Chunk(
                        title=note_title,
                        content=pre_content,
                        note_path=note_path,
                        heading_path=[],
                        metadata=normalized_metadata,
                        start_line=1,
                        end_line=first_heading_line,
                    ))

            # Process each heading section
            heading_stack = []  # [(level, text), ...]

            for idx, (line_num, level, heading_text) in enumerate(headings):
                # Determine end of this section
                if idx + 1 < len(headings):
                    end_line = headings[idx + 1][0]
                else:
                    end_line = len(lines)

                # Update heading stack
                # Pop headings at same or deeper level
                while heading_stack and heading_stack[-1][0] >= level:
                    heading_stack.pop()
                heading_stack.append((level, heading_text))

                # Build heading path
                heading_path = [h[1] for h in heading_stack]

                # Build hierarchical title
                title = note_title + " > " + " > ".join(heading_path)

                # Extract content (excluding the heading line itself)
                section_lines = lines[line_num + 1:end_line]
                section_content = '\n'.join(section_lines).strip()

                if len(section_content) >= self.min_chunk_chars:
                    chunks.append(Chunk(
                        title=title,
                        content=section_content,
                        note_path=note_path,
                        heading_path=heading_path,
                        metadata=normalized_metadata,
                        start_line=line_num + 1,
                        end_line=end_line,
                    ))

        return ChunkedDocument(
            note_path=note_path,
            note_title=note_title,
            chunks=chunks,
            metadata=normalized_metadata,
        )

    def _extract_tags(self, metadata: dict) -> list[str]:
        """Extract tags from frontmatter, handling various formats."""
        tags = metadata.get("tags", [])
        if isinstance(tags, str):
            # Handle comma-separated or space-separated
            tags = [t.strip() for t in re.split(r'[,\s]+', tags) if t.strip()]
        elif not isinstance(tags, list):
            tags = []
        return tags


class VaultChunker:
    """Chunks an entire Obsidian vault."""

    def __init__(
        self,
        vault_path: Path,
        chunker: Optional[MarkdownChunker] = None,
        exclude_patterns: Optional[list[str]] = None,
    ):
        self.vault_path = Path(vault_path)
        self.chunker = chunker or MarkdownChunker()
        self.exclude_patterns = exclude_patterns or [
            ".obsidian",
            ".trash",
            ".git",
            "node_modules",
        ]

    def should_exclude(self, path: Path) -> bool:
        """Check if path should be excluded."""
        path_str = str(path)
        for pattern in self.exclude_patterns:
            if pattern in path_str:
                return True
        return False

    def chunk_vault(self) -> list[ChunkedDocument]:
        """Chunk all markdown files in the vault."""
        documents = []

        for md_file in self.vault_path.rglob("*.md"):
            if self.should_exclude(md_file):
                continue

            try:
                doc = self.chunker.chunk_file(md_file, self.vault_path)
                if doc.chunks:  # Only include if has valid chunks
                    documents.append(doc)
            except Exception as e:
                print(f"Error chunking {md_file}: {e}")

        return documents

    def chunk_vault_to_json(self, output_path: Path) -> dict:
        """Chunk vault and save to JSON."""
        documents = self.chunk_vault()

        # Build output structure
        output = {
            "vault_path": str(self.vault_path),
            "total_documents": len(documents),
            "total_chunks": sum(len(d.chunks) for d in documents),
            "documents": [d.to_dict() for d in documents],
        }

        # Also build flat chunk index for easy lookup
        chunk_index = []
        for doc in documents:
            for chunk in doc.chunks:
                chunk_index.append({
                    "id": f"{doc.note_path}::{'>'.join(chunk.heading_path) or 'root'}",
                    **chunk.to_dict(),
                })

        output["chunk_index"] = chunk_index

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False, cls=DateTimeEncoder)

        return output


def main():
    """CLI for chunking a vault."""
    import argparse

    parser = argparse.ArgumentParser(description="Chunk an Obsidian vault for O-RAG")
    parser.add_argument("vault_path", type=Path, help="Path to Obsidian vault")
    parser.add_argument("-o", "--output", type=Path, default=Path("chunks.json"),
                        help="Output JSON file path")
    parser.add_argument("--min-chars", type=int, default=50,
                        help="Minimum characters per chunk")
    parser.add_argument("--exclude", nargs="*", default=[],
                        help="Additional patterns to exclude")

    args = parser.parse_args()

    chunker = MarkdownChunker(min_chunk_chars=args.min_chars)
    vault_chunker = VaultChunker(
        args.vault_path,
        chunker=chunker,
        exclude_patterns=[".obsidian", ".trash", ".git", "node_modules"] + args.exclude,
    )

    print(f"Chunking vault: {args.vault_path}")
    result = vault_chunker.chunk_vault_to_json(args.output)

    print(f"Done! {result['total_documents']} documents, {result['total_chunks']} chunks")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
