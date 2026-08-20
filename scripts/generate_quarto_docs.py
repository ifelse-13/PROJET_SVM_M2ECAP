#!/usr/bin/env python3
"""Generate mirrored Quarto reference pages for every documentable file in a repository.

For each text file found under *source_root* the script creates a ``.qmd`` page
that contains a short summary, key symbols extracted from the source, and an
excerpt of the raw content.  Directory index pages are also produced so the
generated documentation tree mirrors the original directory structure.

Typical usage::

    python scripts/generate_quarto_docs.py
    python scripts/generate_quarto_docs.py --source-root . --docs-root docs/reference
    python scripts/generate_quarto_docs.py --clean
"""
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
    """Immutable record that holds all documentation data for a single source file.

    Attributes:
        source_path: Path of the source file relative to the repository root.
        output_path: Absolute path of the ``.qmd`` file to be written.
        title: Human-readable display title (typically the file name).
        summary: One-sentence description of the file's purpose or content.
        key_elements: Bullet-point strings listing imports, classes, functions, etc.
        excerpt: Optional fenced-code-block with the first lines of the file.
        language: Lowercase language identifier used in fenced code blocks.
    """

    source_path: Path
    output_path: Path
    title: str
    summary: str
    key_elements: list[str]
    excerpt: str | None
    language: str


def slugify(value: str) -> str:
    """Convert *value* to a URL-safe, lowercase slug.

    Non-ASCII characters are first normalised to their closest ASCII
    equivalent (NFKD + ASCII encoding).  Any remaining character that is not
    alphanumeric, a dot, a hyphen, or an underscore is replaced by a hyphen.
    Leading and trailing hyphens, underscores, and dots are stripped.  If the
    result is empty the string ``"index"`` is returned.

    Args:
        value: Arbitrary string to slugify.

    Returns:
        URL-safe slug string.
    """
    normalized = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", normalized).strip("-_.").lower()
    return slug or "index"


def is_binary_path(path: Path) -> bool:
    """Return ``True`` if *path* should be treated as a binary (non-text) file.

    A file is considered binary when any of the following is true:

    * Its suffix appears in :data:`BINARY_SUFFIXES`.
    * Reading the first :data:`TEXT_SAMPLE_BYTES` bytes raises an
      :exc:`OSError`.
    * The sample contains a null byte ``\\x00``.
    * The sample cannot be decoded as UTF-8.

    An empty file is **not** considered binary.

    Args:
        path: File system path to inspect.

    Returns:
        ``True`` if the file is binary, ``False`` otherwise.
    """
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
    """Read *path* as UTF-8 text, replacing undecodable bytes.

    Args:
        path: File to read.

    Returns:
        Full file contents as a string.
    """
    return path.read_text(encoding="utf-8", errors="replace")


def qmd_name_for(source_path: Path) -> str:
    """Return the ``.qmd`` file name for *source_path*.

    The file name (not the full path) is slugified and the ``.qmd`` extension
    is appended, e.g. ``My Script.py`` → ``my-script.py.qmd``.

    Args:
        source_path: Path whose name should be converted.

    Returns:
        Slugified file name with ``.qmd`` appended.
    """
    return f"{slugify(source_path.name)}.qmd"


def output_path_for(source_path: Path, docs_root: Path) -> Path:
    """Compute the output ``.qmd`` path for a given *source_path*.

    Each directory component of *source_path* is slugified to build a
    mirror directory tree under *docs_root*.

    Args:
        source_path: Relative path of the source file from the repository root.
        docs_root: Absolute path of the root directory for generated docs.

    Returns:
        Absolute path where the ``.qmd`` file for this source should be written.
    """
    parent_parts = [slugify(part) for part in source_path.parent.parts if part not in ("", ".")]
    return docs_root.joinpath(*parent_parts, qmd_name_for(source_path))


def detect_language(path: Path) -> str:
    """Return the language identifier for *path* based on its suffix.

    The identifier is looked up in :data:`LANGUAGE_BY_SUFFIX`.  If the suffix
    is not recognised, ``"text"`` is returned as a safe default.

    Args:
        path: File whose extension should be used for detection.

    Returns:
        Lowercase language string suitable for use in fenced code blocks.
    """
    return LANGUAGE_BY_SUFFIX.get(path.suffix.lower(), "text")


def summarize_requirements(text: str) -> tuple[str, list[str]]:
    """Produce a summary and key-element list for a ``requirements.txt`` file.

    Comment lines (starting with ``#``) and blank lines are ignored when
    counting packages.  Up to eight package names are listed in the
    ``"dependencies"`` key element; a trailing ellipsis is added when there
    are more.

    Args:
        text: Full text content of the requirements file.

    Returns:
        A ``(summary, key_elements)`` tuple where *summary* is a one-sentence
        string and *key_elements* is a list of bullet-point strings.
    """
    packages = [line.strip() for line in text.splitlines() if line.strip() and not line.lstrip().startswith("#")]
    summary = f"Pinned Python dependency list containing {len(packages)} package entries."
    key_elements = [f"dependencies: {', '.join(packages[:8])}{'…' if len(packages) > 8 else ''}"]
    return summary, key_elements


def summarize_markdown(text: str) -> tuple[str, list[str]]:
    """Produce a summary and key-element list for a Markdown or Quarto file.

    The first non-heading, non-image, non-HTML paragraph line is used as the
    summary.  Up to six ATX headings and six image references are reported as
    key elements.

    Args:
        text: Full text content of the Markdown file.

    Returns:
        A ``(summary, key_elements)`` tuple.
    """
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
    """Produce a summary and key-element list for a Python source file.

    The module-level docstring (if any) is used as the summary.  When no
    docstring is present, a synthetic sentence is built from the names of the
    top-level classes and functions found in the AST.  Imports, class names,
    and function names are each reported as a separate key element (up to
    eight items each).

    A :exc:`SyntaxError` during parsing results in a generic summary and an
    empty key-element list rather than raising.

    Args:
        text: Python source code as a string.

    Returns:
        A ``(summary, key_elements)`` tuple.
    """
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
    """Concatenate code-cell sources from *notebook* up to the excerpt limit.

    Code cells are joined with a blank line between them.  Extraction stops
    early once the combined text exceeds :data:`EXCERPT_CHAR_LIMIT` to avoid
    reading unnecessarily large notebooks.

    Args:
        notebook: Parsed Jupyter notebook dictionary.

    Returns:
        Concatenated code snippets as a single string.  Empty string when
        there are no code cells.
    """
    snippets: list[str] = []
    for cell in notebook.get("cells", []):
        if cell.get("cell_type") == "code":
            source = "".join(cell.get("source", []))
            if source.strip():
                snippets.append(source)
        if len("\n\n".join(snippets)) >= EXCERPT_CHAR_LIMIT:
            break
    return "\n\n".join(snippets)


def sanitize_python_snippet(text: str) -> str:
    """Remove IPython magic commands and shell/package-manager invocations.

    Lines that start with ``!``, ``%``, ``pip``, ``pip3``,
    ``python -m pip``, or ``conda`` are dropped so that the remaining code
    can be safely parsed as plain Python.

    Args:
        text: Raw Python or notebook code snippet.

    Returns:
        Cleaned snippet with magic/shell lines removed.
    """
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
    """Produce a summary and key-element list for a Jupyter notebook.

    The summary reports the number of code and markdown cells.  Key elements
    are derived by calling :func:`summarize_python` on the concatenated
    sanitized code cells, plus a list of ATX headings from markdown cells.

    If *text* cannot be decoded as JSON the function returns a generic
    summary, an empty key-element list, and ``None`` for the excerpt source.

    Args:
        text: Raw JSON text of the ``.ipynb`` file.

    Returns:
        A ``(summary, key_elements, code_excerpt)`` triple where
        *code_excerpt* is the raw code concatenated from code cells, or
        ``None`` when no code is present.
    """
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
    return summary, key_elements, code or None


def summarize_structured_text(text: str) -> tuple[str, list[str]]:
    """Produce a summary and key-element list for structured text files.

    This is used for YAML, TOML, JSON, and similar configuration formats.
    The first non-empty line is used as the summary, and top-level key names
    (patterns ``KEY:`` or ``KEY=``) are reported as a key element.

    Args:
        text: Full text content of the structured file.

    Returns:
        A ``(summary, key_elements)`` tuple.
    """
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    summary = lines[0] if lines else "Structured text configuration file."
    keys = re.findall(r"^([A-Za-z0-9_.-]+)\s*[:=]", text, flags=re.MULTILINE)
    key_elements = ["keys: " + ", ".join(keys[:10])] if keys else []
    return summary, key_elements


def summarize_generic_text(text: str) -> tuple[str, list[str]]:
    """Produce a minimal summary for unrecognised plain-text files.

    The summary is the first non-empty line truncated to 200 characters.
    No key elements are extracted.

    Args:
        text: Full text content of the file.

    Returns:
        A ``(summary, key_elements)`` tuple where *key_elements* is always
        an empty list.
    """
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    summary = lines[0][:200] if lines else "Text file."
    return summary, []


def build_excerpt(text: str, language: str) -> str | None:
    """Build a fenced-code-block excerpt from *text*.

    At most :data:`EXCERPT_LINE_LIMIT` lines are included.  If the resulting
    text exceeds :data:`EXCERPT_CHAR_LIMIT` characters it is hard-truncated
    and ``"\\n..."`` is appended to signal truncation.  Empty text yields
    ``None``.

    Args:
        text: Source text to excerpt.
        language: Language identifier for the fenced code fence (e.g.
            ``"python"``).

    Returns:
        A Markdown fenced-code string, or ``None`` if *text* is empty.
    """
    lines = text.splitlines()
    excerpt = "\n".join(lines[:EXCERPT_LINE_LIMIT]).strip()
    if len(excerpt) > EXCERPT_CHAR_LIMIT:
        excerpt = excerpt[:EXCERPT_CHAR_LIMIT].rstrip() + "\n..."
    if not excerpt:
        return None
    return f"```{language}\n{excerpt}\n```"


def describe_file(source_root: Path, path: Path, docs_root: Path) -> FileDoc:
    """Build a :class:`FileDoc` by inspecting the contents of *path*.

    The function dispatches to the appropriate ``summarize_*`` helper based on
    the file name and suffix.  It then constructs the output path, generates
    an excerpt, and returns an immutable :class:`FileDoc` record.

    Args:
        source_root: Absolute path of the repository root used to derive
            the relative source path.
        path: Absolute path of the source file to describe.
        docs_root: Absolute path of the root directory for generated docs,
            used to compute the output path.

    Returns:
        A fully populated :class:`FileDoc` instance.
    """
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
        title=path.name,
        summary=summary,
        key_elements=key_elements,
        excerpt=build_excerpt(excerpt_source, language),
        language=language,
    )


def render_file_doc(file_doc: FileDoc) -> str:
    """Render a :class:`FileDoc` as a Quarto Markdown (``.qmd``) document.

    The output always contains a YAML front-matter block, a level-1 heading,
    a source-path link, and a *Summary* section.  Optional *Key symbols or
    elements* and *Excerpt* sections are appended when the corresponding
    fields are non-empty.

    Args:
        file_doc: Populated documentation record for a single source file.

    Returns:
        Complete ``.qmd`` document as a string, terminated by a newline.
    """
    lines = [
        "---",
        f'title: "{file_doc.source_path.as_posix()}"',
        "---",
        "",
        f"# `{file_doc.source_path.as_posix()}`",
        "",
        f"- Source path: `{file_doc.source_path.as_posix()}`",
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
    """Write *content* to *path* only when the existing content differs.

    Parent directories are created automatically.  This avoids updating file
    modification timestamps (and triggering incremental build tools) when the
    generated content has not changed.

    Args:
        path: Destination file path (need not exist yet).
        content: UTF-8 text to write.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_text(encoding="utf-8") == content:
        return
    path.write_text(content, encoding="utf-8")


def docs_dir_for(source_dir: Path, docs_root: Path) -> Path:
    """Return the output directory that mirrors *source_dir* under *docs_root*.

    Each component of *source_dir* is slugified.  The repository root
    (represented as ``Path(".")``) maps directly to *docs_root*.

    Args:
        source_dir: Relative directory path from the repository root.
        docs_root: Absolute path of the root directory for generated docs.

    Returns:
        Absolute path of the corresponding directory under *docs_root*.
    """
    if source_dir == Path("."):
        return docs_root
    return docs_root.joinpath(*[slugify(part) for part in source_dir.parts])


def render_directory_index(source_dir: Path, docs_root: Path, child_dirs: list[Path], file_docs: list[FileDoc]) -> str:
    """Render a directory-level ``index.qmd`` listing sub-directories and files.

    Subdirectory entries link to their own ``index.qmd`` using paths relative
    to the current directory's output location.  File entries link to the
    individual ``.qmd`` page produced for that file.  When there are no
    entries a placeholder message is rendered instead.

    Args:
        source_dir: Relative path of the directory being documented.
        docs_root: Absolute path of the root directory for generated docs.
        child_dirs: Sorted list of relative paths for immediate subdirectories.
        file_docs: List of :class:`FileDoc` instances for files in this
            directory.

    Returns:
        Complete ``index.qmd`` document as a string, terminated by a newline.
    """
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
    """Yield every documentable (non-binary, non-ignored) file under *source_root*.

    Directories listed in :data:`IGNORED_DIR_NAMES` are pruned during the
    walk.  The ``docs`` output tree itself is also excluded to prevent
    self-referential documentation.  Binary files detected by
    :func:`is_binary_path` and files in :data:`IGNORED_FILE_NAMES` are
    skipped.  Files within each directory are yielded in alphabetical order.

    Args:
        source_root: Absolute path of the repository root to walk.
        docs_root: Absolute path of the root directory for generated docs.
            The parent of this directory is used as the exclusion boundary.

    Yields:
        Absolute :class:`~pathlib.Path` objects for each documentable file.
    """
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
    """Remove ``.qmd`` files and empty directories that are no longer expected.

    Any ``.qmd`` file under *docs_root* that is not present in
    *expected_paths* is deleted.  Empty directories left behind after deletion
    are also removed (deepest first to allow cascading removal).

    Args:
        docs_root: Root of the generated docs tree to clean.
        expected_paths: Set of absolute paths that should be retained.
    """
    if not docs_root.exists():
        return
    for path in sorted(docs_root.rglob("*.qmd"), reverse=True):
        if path not in expected_paths:
            path.unlink()
    for path in sorted(docs_root.rglob("*"), reverse=True):
        if path.is_dir() and not any(path.iterdir()):
            path.rmdir()


def generate_docs(source_root: Path, docs_root: Path) -> list[Path]:
    """Scan *source_root* and write Quarto documentation under *docs_root*.

    The function walks the repository, describes each documentable file,
    writes individual ``.qmd`` pages and directory ``index.qmd`` files, then
    removes any stale output files left from previous runs.

    Args:
        source_root: Absolute path of the repository root to document.
        docs_root: Absolute path of the destination directory for generated
            ``.qmd`` files.

    Returns:
        Sorted list of absolute paths of all ``.qmd`` files written or
        updated during this run.
    """
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
    """Parse command-line arguments for the documentation generator.

    Defines three optional arguments:

    * ``--source-root``: repository root to scan (defaults to the parent of
      the ``scripts/`` directory).
    * ``--docs-root``: destination directory for generated ``.qmd`` files
      (defaults to ``<source-root>/docs/reference``).
    * ``--clean``: when set, the docs root is deleted before regenerating.

    Returns:
        Parsed :class:`argparse.Namespace` with attributes ``source_root``,
        ``docs_root``, and ``clean``.
    """
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
    """Entry point: parse arguments, run the generator, and print a summary.

    If ``--clean`` is passed and the docs root exists, it is removed entirely
    before generation begins.  After generation, the number of produced files
    and the output directory are printed to standard output.

    Returns:
        Exit code ``0`` on success.
    """
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
