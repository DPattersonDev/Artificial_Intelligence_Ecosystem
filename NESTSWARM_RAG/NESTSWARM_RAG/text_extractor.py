import argparse
from pathlib import Path
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup
from pypdf import PdfReader


def extract_from_web(url: str) -> str:
    """Download a web page and return readable text."""
    response = requests.get(
        url,
        timeout=30,
        headers={"User-Agent": "Mozilla/5.0"},
    )
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")

    for tag in soup(["script", "style", "nav", "footer", "header", "noscript"]):
        tag.decompose()

    lines = [line.strip() for line in soup.stripped_strings]
    return "\n".join(line for line in lines if line)


def extract_from_pdf(path: Path) -> str:
    """Extract text from every page of a PDF."""
    reader = PdfReader(str(path))
    pages = []

    for page in reader.pages:
        text = page.extract_text() or ""
        if text.strip():
            pages.append(text.strip())

    return "\n\n".join(pages)


def extract_from_text(path: Path) -> str:
    """Read an existing UTF-8 text file."""
    return path.read_text(encoding="utf-8")


def looks_like_url(value: str) -> bool:
    parsed = urlparse(value)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def main():
    parser = argparse.ArgumentParser(
        description="Extract text from a web page, PDF, or text file."
    )
    parser.add_argument(
        "source",
        help="URL or path to a .pdf/.txt source",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="Selected_Document.txt",
        help="Output text file (default: Selected_Document.txt)",
    )
    args = parser.parse_args()

    if looks_like_url(args.source):
        text = extract_from_web(args.source)
    else:
        source_path = Path(args.source)

        if not source_path.exists():
            raise FileNotFoundError(f"Source not found: {source_path}")

        suffix = source_path.suffix.lower()

        if suffix == ".pdf":
            text = extract_from_pdf(source_path)
        elif suffix in {".txt", ".md"}:
            text = extract_from_text(source_path)
        else:
            raise ValueError("Supported local file types are .pdf, .txt, and .md")

    text = text.strip()

    if not text:
        raise ValueError("No readable text was extracted from the source.")

    output_path = Path(args.output)
    output_path.write_text(text + "\n", encoding="utf-8")

    print(f"Saved extracted text to: {output_path}")
    print(f"Characters extracted: {len(text):,}")


if __name__ == "__main__":
    main()
