#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
import os
import re
import shutil
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


IGNORED_DIR_NAMES = {
    ".git",
    ".hg",
    ".idea",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
    ".venv",
    "__pycache__",
    "_site",
    "build",
    "dist",
    "node_modules",
    "site-packages",
    "target",
    "vendor",
    "venv",
}

IGNORED_FILE_NAMES = {".DS_Store", "desktop.ini"}
TEXT_SAMPLE_BYTES = 8192
EXCERPT_CHAR_LIMIT = 1400
EXCERPT_LINE_LIMIT = 40

BINARY_SUFFIXES = {
    ".7z",
    ".bin",
    ".bmp",
    ".class",
    ".dll",
    ".dylib",
    ".exe",
    ".gif",
    ".gz",
    ".ico",
    ".jpeg",
    ".jpg",
    ".mov",
    ".mp3",
    ".mp4",
    ".o",
    ".pdf",
    ".pickle",
    ".png",
    ".pyc",
    ".pyd",
    ".pyo",
    ".so",
    ".tar",
    ".tif",
    ".tiff",
    ".wav",
    ".webp",
    ".zip",
}

LANGUAGE_BY_SUFFIX = {
    ".css": "css",
    ".ipynb": "python",
    ".js": "javascript",
    ".json": "json",
    ".md": "markdown",
    ".py": "python",
    ".qmd": "markdown",
    ".r": "r",
    ".rs": "rust",
    ".sh": "bash",
    ".sql": "sql",
    ".toml": "toml",
    ".txt": "text",
    ".yaml": "yaml",
    ".yml": "yaml",
}


@dataclass(frozen=True)
class FileDoc:
    source_path: Path
    output_path: Path
    doc_path: Path
    title: str
    summary: str
    key_elements: list[str]
    excerpt: str | None
    language: str


def slugify(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", normalized).strip("-_.").lower()
    return slug or "index"


def is_binary_path(path: Path) -> bool:
    if path.suffix.lower() in BINARY_SUFFIXES:
        return True
    try:
        sample = path.read_bytes()[:TEXT_SAMPLE_BYTES]
    except OSError:
        return True
    if b"\x00" in sample:
        return True
    if not sample:
        return False
    try:
        sample.decode("utf-8")
    except UnicodeDecodeError:
        return True
    return False


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def qmd_name_for(source_path: Path) -> str:
    return f"{slugify(source_path.name)}.qmd"


def output_path_for(source_path: Path, docs_root: Path) -> Path:
    parent_parts = [slugify(part) for part in source_path.parent.parts if part not in ("", ".")]
    return docs_root.joinpath(*parent_parts, qmd_name_for(source_path))


def detect_language(path: Path) -> str:
    return LANGUAGE_BY_SUFFIX.get(path.suffix.lower(), "text")


def summarize_requirements(text: str) -> tuple[str, list[str]]:
    packages = [line.strip() for line in text.splitlines() if line.strip() and not line.lstrip().startswith("#")]
    summary = f"Pinned Python dependency list containing {len(packages)} package entries."
    key_elements = [f"dependencies: {', '.join(packages[:8])}{'…' if len(packages) > 8 else ''}"]
    return summary, key_elements


def summarize_markdown(text: str) -> tuple[str, list[str]]:
    headings = [line.lstrip("# ").strip() for line in text.splitlines() if line.startswith("#")]
    paragraphs = [line.strip() for line in text.splitlines() if line.strip() and not line.startswith(("#", "![", "<"))]
    summary = paragraphs[0] if paragraphs else "Markdown document describing project context and results."
    key_elements: list[str] = []
    if headings:
        key_elements.append("headings: " + ", ".join(headings[:6]))
    image_refs = re.findall(r"!\[[^\]]*\]\(([^)]+)\)", text)
    if image_refs:
        key_elements.append("linked assets: " + ", ".join(image_refs[:6]))
    return summary, key_elements


def summarize_python(text: str) -> tuple[str, list[str]]:
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return "Python source file with syntax that could not be fully parsed for symbols.", []

    imports: list[str] = []
    functions: list[str] = []
    classes: list[str] = []
    docstring = ast.get_docstring(tree)

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names[:5])
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            imports.extend(f"{module}.{alias.name}".strip(".") for alias in node.names[:5])
        elif isinstance(node, ast.FunctionDef):
            functions.append(node.name)
        elif isinstance(node, ast.AsyncFunctionDef):
            functions.append(node.name)
        elif isinstance(node, ast.ClassDef):
            classes.append(node.name)

    if docstring:
        summary = docstring.splitlines()[0].strip()
    elif functions or classes:
        summary = "Python module defining " + ", ".join((classes + functions)[:4]) + "."
    elif imports:
        summary = "Python module primarily composed of imports and top-level statements."
    else:
        summary = "Python source file with executable top-level logic."

    key_elements: list[str] = []
    if imports:
        key_elements.append("imports: " + ", ".join(sorted(dict.fromkeys(imports))[:8]))
    if classes:
        key_elements.append("classes: " + ", ".join(classes[:8]))
    if functions:
        key_elements.append("functions: " + ", ".join(functions[:8]))
    return summary, key_elements


def extract_notebook_code(notebook: dict) -> str:
    snippets: list[str] = []
    snippet_length = 0
    for cell in notebook.get("cells", []):
        if cell.get("cell_type") == "code":
            source = "".join(cell.get("source", []))
            if source.strip():
                snippets.append(source)
                snippet_length += len(source) + 2
                if snippet_length >= EXCERPT_CHAR_LIMIT:
                    break
    return "\n\n".join(snippets)


def sanitize_python_snippet(text: str) -> str:
    cleaned_lines: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith(("!", "%")):
            continue
        if stripped.startswith(("pip ", "pip3 ", "python -m pip ", "conda ")):
            continue
        cleaned_lines.append(line)
    return "\n".join(cleaned_lines)


def summarize_notebook(text: str) -> tuple[str, list[str], str | None]:
    try:
        notebook = json.loads(text)
    except json.JSONDecodeError:
        return "Notebook file that could not be decoded as JSON.", [], None

    code_cells = sum(1 for cell in notebook.get("cells", []) if cell.get("cell_type") == "code")
    markdown_cells = sum(1 for cell in notebook.get("cells", []) if cell.get("cell_type") == "markdown")
    summary = f"Jupyter notebook with {code_cells} code cells and {markdown_cells} markdown cells."

    code = extract_notebook_code(notebook)
    sanitized_code = sanitize_python_snippet(code)
    key_elements: list[str] = []
    if sanitized_code.strip():
        _, detected = summarize_python(sanitized_code)
        key_elements.extend(detected)

    headings: list[str] = []
    for cell in notebook.get("cells", []):
        if cell.get("cell_type") == "markdown":
            for line in "".join(cell.get("source", [])).splitlines():
                if line.startswith("#"):
                    headings.append(line.lstrip("# ").strip())
    if headings:
        key_elements.append("markdown headings: " + ", ".join(headings[:6]))
    return summary, key_elements, sanitized_code or None


def summarize_structured_text(text: str) -> tuple[str, list[str]]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    summary = lines[0] if lines else "Structured text configuration file."
    keys = re.findall(r"^([A-Za-z0-9_.-]+)\s*[:=]", text, flags=re.MULTILINE)
    key_elements = ["keys: " + ", ".join(keys[:10])] if keys else []
    return summary, key_elements


def summarize_generic_text(text: str) -> tuple[str, list[str]]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    summary = lines[0][:200] if lines else "Text file."
    return summary, []


def build_excerpt(text: str, language: str) -> str | None:
    lines = text.splitlines()
    excerpt = "\n".join(lines[:EXCERPT_LINE_LIMIT]).strip()
    if len(excerpt) > EXCERPT_CHAR_LIMIT:
        excerpt = excerpt[:EXCERPT_CHAR_LIMIT].rstrip() + "\n..."
    if not excerpt:
        return None
    return f"```{language}\n{excerpt}\n```"


def describe_file(source_root: Path, path: Path, docs_root: Path) -> FileDoc:
    relative_path = path.relative_to(source_root)
    language = detect_language(path)
    text = read_text(path)

    if path.name == "requirements.txt":
        summary, key_elements = summarize_requirements(text)
        excerpt_source = text
    elif path.suffix.lower() == ".ipynb":
        summary, key_elements, notebook_excerpt = summarize_notebook(text)
        excerpt_source = notebook_excerpt or text
    elif path.suffix.lower() == ".py":
        summary, key_elements = summarize_python(text)
        excerpt_source = text
    elif path.suffix.lower() in {".md", ".qmd"}:
        summary, key_elements = summarize_markdown(text)
        excerpt_source = text
    elif path.suffix.lower() in {".yml", ".yaml", ".toml", ".json"}:
        summary, key_elements = summarize_structured_text(text)
        excerpt_source = text
    else:
        summary, key_elements = summarize_generic_text(text)
        excerpt_source = text

    return FileDoc(
        source_path=relative_path,
        output_path=output_path_for(relative_path, docs_root),
        doc_path=output_path_for(relative_path, docs_root).relative_to(docs_root.parent),
        title=path.name,
        summary=summary,
        key_elements=key_elements,
        excerpt=build_excerpt(excerpt_source, language),
        language=language,
    )


def render_file_doc(file_doc: FileDoc) -> str:
    lines = [
        "---",
        f'title: "{file_doc.source_path.as_posix()}"',
        "---",
        "",
        f"# `{file_doc.source_path.as_posix()}`",
        "",
        f"- Source path: `{file_doc.source_path.as_posix()}`",
        f"- Documentation path: `{file_doc.doc_path.as_posix()}`",
        "",
        "## Summary",
        "",
        file_doc.summary,
        "",
    ]

    if file_doc.key_elements:
        lines.extend(["## Key symbols or elements", ""])
        lines.extend(f"- {element}" for element in file_doc.key_elements)
        lines.append("")

    if file_doc.excerpt:
        lines.extend(["## Excerpt", "", file_doc.excerpt, ""])

    return "\n".join(lines).rstrip() + "\n"


def write_if_changed(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_text(encoding="utf-8") == content:
        return
    path.write_text(content, encoding="utf-8")


def docs_dir_for(source_dir: Path, docs_root: Path) -> Path:
    if source_dir == Path("."):
        return docs_root
    return docs_root.joinpath(*[slugify(part) for part in source_dir.parts])


def render_directory_index(source_dir: Path, docs_root: Path, child_dirs: list[Path], file_docs: list[FileDoc]) -> str:
    source_label = source_dir.as_posix() if source_dir.parts else "."
    current_docs_dir = docs_dir_for(source_dir, docs_root)
    lines = [
        "---",
        f'title: "Reference: {source_label}"',
        "---",
        "",
        f"# Reference: `{source_label}`",
        "",
        "## Contents",
        "",
    ]

    if child_dirs:
        lines.append("### Directories")
        lines.append("")
        for child_dir in sorted(child_dirs):
            label = child_dir.as_posix()
            target = docs_dir_for(child_dir, docs_root).joinpath("index.qmd").relative_to(current_docs_dir).as_posix()
            lines.append(f"- [{label}]({target})")
        lines.append("")

    if file_docs:
        lines.append("### Files")
        lines.append("")
        for file_doc in sorted(file_docs, key=lambda item: item.source_path.as_posix()):
            lines.append(f"- [`{file_doc.source_path.name}`]({qmd_name_for(file_doc.source_path)})")
        lines.append("")

    if not child_dirs and not file_docs:
        lines.extend(["_No documentable files found in this directory._", ""])

    return "\n".join(lines).rstrip() + "\n"


def iter_source_files(source_root: Path, docs_root: Path) -> Iterable[Path]:
    docs_project_root = docs_root.parent
    for current_root, dir_names, file_names in os.walk(source_root, topdown=True):
        current_path = Path(current_root)
        relative_dir = current_path.relative_to(source_root)
        if current_path == docs_project_root or docs_project_root in current_path.parents:
            dir_names[:] = []
            continue

        filtered_dir_names: list[str] = []
        for dir_name in dir_names:
            candidate_dir = current_path / dir_name
            if dir_name in IGNORED_DIR_NAMES:
                continue
            if candidate_dir == docs_project_root:
                continue
            filtered_dir_names.append(dir_name)
        dir_names[:] = filtered_dir_names

        for file_name in sorted(file_names):
            if file_name in IGNORED_FILE_NAMES:
                continue
            candidate = current_path / file_name
            if is_binary_path(candidate):
                continue
            yield source_root / relative_dir / file_name


def cleanup_stale_outputs(docs_root: Path, expected_paths: set[Path]) -> None:
    if not docs_root.exists():
        return
    for path in sorted(docs_root.rglob("*.qmd"), reverse=True):
        if path not in expected_paths:
            path.unlink()
    for path in sorted(docs_root.rglob("*"), reverse=True):
        if path.is_dir() and not any(path.iterdir()):
            path.rmdir()


def generate_docs(source_root: Path, docs_root: Path) -> list[Path]:
    file_docs: list[FileDoc] = []
    directory_map: dict[Path, list[FileDoc]] = {}
    child_dir_map: dict[Path, set[Path]] = {}

    for source_file in iter_source_files(source_root, docs_root):
        file_doc = describe_file(source_root, source_file, docs_root)
        file_docs.append(file_doc)
        directory_map.setdefault(file_doc.source_path.parent, []).append(file_doc)
        directory = file_doc.source_path.parent
        while directory.parts:
            parent = directory.parent
            child_dir_map.setdefault(parent, set()).add(directory)
            directory = parent
        child_dir_map.setdefault(Path("."), set())

    expected_paths: set[Path] = set()
    for file_doc in file_docs:
        write_if_changed(file_doc.output_path, render_file_doc(file_doc))
        expected_paths.add(file_doc.output_path)

    directories = set(directory_map) | set(child_dir_map) | {Path(".")}
    for source_dir in sorted(directories, key=lambda path: (len(path.parts), path.as_posix())):
        index_path = docs_dir_for(source_dir, docs_root) / "index.qmd"
        content = render_directory_index(
            source_dir,
            docs_root,
            sorted(child_dir_map.get(source_dir, set())),
            directory_map.get(source_dir, []),
        )
        write_if_changed(index_path, content)
        expected_paths.add(index_path)

    cleanup_stale_outputs(docs_root, expected_paths)
    return sorted(expected_paths)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate mirrored Quarto reference pages for repository files.")
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Repository root to scan.",
    )
    parser.add_argument(
        "--docs-root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "docs" / "reference",
        help="Destination directory for generated `.qmd` files.",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove the generated docs directory before regenerating.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source_root = args.source_root.resolve()
    docs_root = args.docs_root.resolve()
    if args.clean and docs_root.exists():
        shutil.rmtree(docs_root)
    generated = generate_docs(source_root, docs_root)
    print(f"Generated {len(generated)} Quarto files in {docs_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
