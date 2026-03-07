# scraper.py
# ScraperCache + PortfolioScraper with Gemini-first extraction for PDFs & Office files
from __future__ import annotations

import collections
import hashlib
import io
import json
import logging
import os
import random
import re
import tempfile
import time
from copy import copy
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from urllib.parse import urljoin, urlparse, urlunparse

import requests
import urllib3
from bs4 import BeautifulSoup, Comment
from google import genai
from google.genai import types

# Silence insecure request warnings when we intentionally fallback to verify=False
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class ScrapedDocument:
    """One logical unit of scraped content ready for chunking."""
    index: int
    title: str
    section: str
    url: str
    content: str
    doc_type: str = "text"  # text | index | video_summary
    extra: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_YT_PATTERN = re.compile(
    r"(?:https?://)?(?:www\.)?(?:youtube\.com/(?:watch\?v=|embed/)|youtu\.be/)([\w\-]{6,})",
    flags=re.I,
)

_SKIP_TAGS = {
    "script", "style", "noscript", "head", "meta", "link",
    "svg", "footer", "iframe", "nav", "aside", "header",
}

_DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept":                    "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language":           "en-US,en;q=0.9",
    "Accept-Encoding":           "gzip, deflate, br",
    "Connection":                "keep-alive",
    "Upgrade-Insecure-Requests": "1",
    "Sec-Fetch-Dest":            "document",
    "Sec-Fetch-Mode":            "navigate",
    "Sec-Fetch-Site":            "none",
}

_SECTION_KEYWORDS: dict[str, str] = {
    "biography": "biography", "bio": "biography", "about": "biography", "profile": "biography", "home": "biography",
    "video": "video", "videos": "video", "talks": "video", "presentations": "video", "youtube": "video", "media": "video",
    "opinion": "opinion", "blog": "opinion", "post": "opinion",
    "employment": "employment", "experience": "employment", "career": "employment", "resume": "employment", "cv": "employment",
    "education": "education", "study": "education", "academic": "education",
    "advisor": "advisor", "advisors": "advisor", "advisory": "advisor",
    "cases": "cases", "case": "cases",
    "research": "research", "publications": "research", "papers": "research",
    "associate": "associate_editor", "editor": "associate_editor", "editorial": "associate_editor",
    "contact": "contact", "email": "contact",
    "executive": "executive_teaching", "teaching": "executive_teaching", "courses": "executive_teaching", "mba": "executive_teaching",
    "progress": "research_in_progress", "ongoing": "research_in_progress",
    "working": "working_papers", "workingpapers": "working_papers", "ssrn": "working_papers", "preprint": "working_papers",
    "fame": "fame", "finance": "fame", "accounting": "fame",
}

# ---------------------------------------------------------------------------
# ScraperCache
# ---------------------------------------------------------------------------

class ScraperCache:
    """
    Layout
    ------
    cache_dir/
      pages/  {md5(url)}.json  -> {url, final_url, html, content_type, cached_at}
      videos/ {md5(url)}.json  -> {url, summary, cached_at}
    """

    def __init__(self, cache_dir: str = "./scraper_cache") -> None:
        self._pages_dir = Path(cache_dir) / "pages"
        self._videos_dir = Path(cache_dir) / "videos"
        self._pages_dir.mkdir(parents=True, exist_ok=True)
        self._videos_dir.mkdir(parents=True, exist_ok=True)

    def get_page(self, url: str) -> Optional[dict]:
        path = self._pages_dir / f"{_url_hash(url)}.json"
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(data, dict) or "html" not in data:
                path.unlink(missing_ok=True)
                return None
            return data
        except Exception as exc:
            logger.warning("cache read error (page) %s: %s", url, exc)
            return None

    def set_page(self, url: str, final_url: str, content: str, content_type: str = "text/html") -> None:
        path = self._pages_dir / f"{_url_hash(url)}.json"
        try:
            path.write_text(json.dumps({
                "url": url,
                "final_url": final_url,
                "html": content,
                "content_type": content_type,
                "cached_at": _now_iso(),
            }, ensure_ascii=False), encoding="utf-8")
        except Exception as exc:
            logger.warning("cache write error (page) %s: %s", url, exc)

    def get_video(self, url: str) -> Optional[str]:
        path = self._videos_dir / f"{_url_hash(url)}.json"
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))["summary"]
        except Exception as exc:
            logger.warning("cache read error (video) %s: %s", url, exc)
            return None

    def set_video(self, url: str, summary: str) -> None:
        path = self._videos_dir / f"{_url_hash(url)}.json"
        try:
            path.write_text(json.dumps({"url": url, "summary": summary, "cached_at": _now_iso()}, ensure_ascii=False),
                            encoding="utf-8")
        except Exception as exc:
            logger.warning("cache write error (video) %s: %s", url, exc)

    def stats(self) -> dict:
        return {
            "cached_pages": len(list(self._pages_dir.glob("*.json"))),
            "cached_videos": len(list(self._videos_dir.glob("*.json"))),
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _url_hash(url: str) -> str:
    return hashlib.md5(url.encode("utf-8")).hexdigest()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _is_youtube(url: str) -> bool:
    return bool(_YT_PATTERN.search(url or ""))


def _is_pdf_url(url: str) -> bool:
    return bool(url and url.lower().split("?")[0].rstrip("/").endswith(".pdf"))


def _is_pdf_ct(ct: str) -> bool:
    return bool(ct and "application/pdf" in ct.lower())


def _is_office_url(url: str) -> bool:
    if not url:
        return False
    low = url.lower().split("?")[0].rstrip("/")
    return any(low.endswith(ext) for ext in (
        ".xlsx", ".xls", ".xlsm", ".xlsb", ".csv",
        ".docx", ".doc", ".pptx", ".ppt", ".odt", ".ods",
    ))


def _is_office_ct(ct: str) -> bool:
    if not ct:
        return False
    ct = ct.lower()
    office_signatures = [
        "application/vnd.ms-excel",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "application/msword",
        "application/vnd.ms-powerpoint",
        "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        "text/csv",
        "application/vnd.oasis.opendocument.spreadsheet",
        "application/vnd.oasis.opendocument.text",
    ]
    return any(sig in ct for sig in office_signatures)


def _clean_text(text: str) -> str:
    text = re.sub(r"\r\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def _strip_noise(soup: BeautifulSoup) -> None:
    for tag in soup.find_all(_SKIP_TAGS):
        try:
            tag.decompose()
        except Exception:
            tag.extract()
    for comment in soup.find_all(string=lambda t: isinstance(t, Comment)):
        comment.extract()


def _extract_plain_text(soup: BeautifulSoup) -> str:
    s = copy(soup)
    _strip_noise(s)
    return _clean_text(s.get_text(separator="\n"))


def _extract_text_with_links(soup: BeautifulSoup, base_url: str) -> str:
    s = copy(soup)
    _strip_noise(s)
    for a_tag in s.find_all("a", href=True):
        try:
            href = urljoin(base_url, a_tag["href"])
        except Exception:
            href = a_tag.get("href", "")
        anchor = a_tag.get_text(strip=True) or href
        a_tag.replace_with(f" [{anchor}]({href}) ")
    return _clean_text(s.get_text(separator="\n"))


def _collect_links(soup: BeautifulSoup, base_url: str) -> list[str]:
    seen: set[str] = set()
    links: list[str] = []
    for a_tag in soup.find_all("a", href=True):
        raw = a_tag["href"].strip()
        if raw.lower().startswith(("javascript:", "mailto:", "tel:", "#", "data:")):
            continue
        try:
            href = urljoin(base_url, raw)
        except Exception:
            continue
        p = urlparse(href)
        if p.scheme not in ("http", "https"):
            continue
        href = urlunparse((p.scheme, p.netloc, p.path, "", p.query, ""))
        if href and href not in seen:
            seen.add(href)
            links.append(href)
    return links


def _url_path_depth(url: str) -> int:
    return len([p for p in urlparse(url).path.strip("/").split("/") if p])


def _first_path_segment(url: str) -> str:
    parts = [p for p in urlparse(url).path.strip("/").split("/") if p]
    return parts[0].lower() if parts else ""


def _infer_section_from_url(url: str, anchor_text: str = "") -> str:
    tokens = re.split(r"[/\-_ .]", (url + " " + anchor_text).lower())
    for token in tokens:
        token = token.strip()
        if token and token in _SECTION_KEYWORDS:
            return _SECTION_KEYWORDS[token]
    return "general"


def _page_title(soup: BeautifulSoup, fallback: str) -> str:
    tag = soup.find("title")
    if tag and tag.string:
        return tag.string.strip()
    h1 = soup.find("h1")
    if h1:
        return h1.get_text(strip=True)
    return fallback


# ---------------------------------------------------------------------------
# PDF helpers (local stages kept as optional fallbacks)
# ---------------------------------------------------------------------------

def _pdf_stage1_pypdf(pdf_bytes: bytes) -> str:
    try:
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(pdf_bytes))
        pages = []
        for page in reader.pages:
            try:
                t = page.extract_text() or ""
                pages.append(t)
            except Exception:
                continue
        return _clean_text("\n\n".join(pages))
    except Exception as exc:
        logger.debug("pypdf error: %s", exc)
        return ""


def _pdf_stage2_pdfminer(pdf_bytes: bytes) -> str:
    try:
        from pdfminer.high_level import extract_text as _pm_extract
        text = _pm_extract(io.BytesIO(pdf_bytes)) or ""
        return _clean_text(text)
    except ImportError:
        return ""
    except Exception as exc:
        logger.debug("pdfminer error: %s", exc)
        return ""


# ---------------------------------------------------------------------------
# Article text extractor — trafilatura
# ---------------------------------------------------------------------------

def _extract_article_text(html: str, url: str = "") -> str:
    try:
        import trafilatura
        text = trafilatura.extract(
            html,
            url=url or None,
            include_comments=False,
            include_tables=True,
            no_fallback=False,
            favor_recall=True,
        )
        return _clean_text(text) if text else ""
    except ImportError:
        return ""
    except Exception as exc:
        logger.debug("trafilatura failed for %s: %s", url, exc)
        return ""


# ---------------------------------------------------------------------------
# PortfolioScraper
# ---------------------------------------------------------------------------

class PortfolioScraper:
    """
    BFS crawler with Gemini-first extraction for binary docs (PDF/Office).
    """

    def __init__(
        self,
        gemini_api_key: str,
        cache: Optional[ScraperCache] = None,
        gemini_model: str = "gemini-2.0-flash",
        request_timeout: int = 20,
        request_delay: float = 0.5,
        follow_external: bool = True,
        max_crawl_pages: int = 200,
    ) -> None:
        self.gemini_model = gemini_model
        self.timeout = request_timeout
        self.delay = request_delay
        self.follow_external = follow_external
        self.max_crawl_pages = max_crawl_pages
        self.cache = cache

        self._session = requests.Session()
        self._session.headers.update(_DEFAULT_HEADERS)
        # Gemini client initialised only when key provided
        try:
            self._gemini = genai.Client(api_key=gemini_api_key) if gemini_api_key else None
        except Exception:
            self._gemini = None

        self._doc_counter: int = 0
        self._visited: set[str] = set()

    # ------------------------------------------------------------------
    # Primary entry point
    # ------------------------------------------------------------------

    def scrape_portfolio(self, root_url: str) -> list[ScrapedDocument]:
        self._reset_session()
        root_url = root_url.rstrip("/")
        root_domain = urlparse(root_url).netloc

        crawl_map: dict[str, dict] = {}
        external_links: dict[str, str] = {}

        self._bfs_crawl(root_url, root_domain, crawl_map, external_links)
        logger.info("BFS done: %d same-domain pages, %d external links", len(crawl_map), len(external_links))

        docs: list[ScrapedDocument] = []
        section_roots: dict[str, str] = {}
        section_children: dict[str, list[str]] = collections.defaultdict(list)

        for url, info in crawl_map.items():
            d, s = info["depth"], info["section"]
            if d == 0:
                docs.append(self._make_index_doc(soup=info["soup"], url=url, section="biography",
                                                 title=_page_title(info["soup"], "Portfolio Home")))
            elif d == 1:
                section_roots[s] = url
            else:
                section_children[s].append(url)

        for section, section_url in section_roots.items():
            info = crawl_map[section_url]
            idx = self._make_index_doc(soup=info["soup"], url=section_url, section=section,
                                       title=f"{section.replace('_', ' ').title()} – Index")
            docs.append(idx)
            logger.info("  [index] %-22s %s  (%d chars)", section, section_url, len(idx.content))
            for child_url in section_children.get(section, []):
                ci = crawl_map[child_url]
                content = _extract_plain_text(ci["soup"])
                if content:
                    docs.append(self._make_doc(title=_page_title(ci["soup"], child_url),
                                               section=section, url=child_url,
                                               content=content, doc_type="text"))

        for section in set(section_children) - set(section_roots):
            for child_url in section_children[section]:
                ci = crawl_map[child_url]
                content = _extract_plain_text(ci["soup"])
                if content:
                    docs.append(self._make_doc(title=_page_title(ci["soup"], child_url),
                                               section=section, url=child_url,
                                               content=content, doc_type="text"))

        if self.follow_external:
            for ext_url, section in external_links.items():
                if ext_url in self._visited:
                    continue
                self._visited.add(ext_url)
                time.sleep(self.delay)
                if _is_youtube(ext_url):
                    docs.extend(self._summarise_video(ext_url, section))
                else:
                    doc = self._scrape_external_page(ext_url, section)
                    if doc:
                        docs.append(doc)

        logger.info("scrape_portfolio done: %d documents", len(docs))
        return docs

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def process_section(self, section_url: str, section_name: str) -> list[ScrapedDocument]:
        if section_url in self._visited:
            return []
        self._visited.add(section_url)
        root_domain = urlparse(section_url).netloc
        soup = self._fetch_soup(section_url)
        if soup is None:
            return []
        docs: list[ScrapedDocument] = [self._make_index_doc(soup=soup, url=section_url, section=section_name,
                                                           title=f"{section_name.replace('_', ' ').title()} – Index")]
        for link in _collect_links(soup, section_url):
            if link in self._visited:
                continue
            is_ext = urlparse(link).netloc != root_domain
            if is_ext and not self.follow_external:
                continue
            self._visited.add(link)
            time.sleep(self.delay)
            if _is_youtube(link):
                docs.extend(self._summarise_video(link, section_name))
            else:
                doc = (self._scrape_external_page(link, section_name)
                       if is_ext else self._scrape_internal_page(link, section_name))
                if doc:
                    docs.append(doc)
        return docs

    def summarise_videos(self, youtube_urls: list[str], section: str = "video") -> list[ScrapedDocument]:
        docs: list[ScrapedDocument] = []
        for url in youtube_urls:
            if url in self._visited:
                continue
            self._visited.add(url)
            docs.extend(self._summarise_video(url, section))
        return docs

    def reset(self) -> None:
        self._reset_session()

    # ------------------------------------------------------------------
    # BFS crawler
    # ------------------------------------------------------------------

    def _bfs_crawl(self, root_url: str, root_domain: str, crawl_map: dict, external_links: dict) -> None:
        queue: collections.deque[tuple[str, int, str, str]] = collections.deque()
        queue.append((root_url, 0, "biography", "Home"))
        self._visited.add(root_url)
        same_domain_count = 0

        while queue:
            url, depth, section, anchor = queue.popleft()

            if same_domain_count >= self.max_crawl_pages:
                logger.warning("BFS hit max_crawl_pages=%d — stopping.", self.max_crawl_pages)
                break

            cached = self.cache.get_page(url) if self.cache else None
            time.sleep(0 if cached else self.delay)

            if cached:
                logger.info("  [page cache HIT] %s", url)
                html = cached["html"]
                final_url = cached.get("final_url", url)
                soup = BeautifulSoup(html, "html.parser")
            else:
                soup, final_url = self._fetch_soup_and_final(url)
                if soup is None:
                    continue

            same_domain_count += 1
            crawl_map[url] = {"soup": soup, "depth": depth, "section": section, "anchor": anchor, "final_url": final_url}
            logger.info("  [BFS depth=%d] %-22s  %s", depth, section, url)

            for link in _collect_links(soup, final_url):
                p = urlparse(link)
                if p.netloc == root_domain:
                    if link in self._visited or link == root_url:
                        continue
                    self._visited.add(link)
                    queue.append((link, _url_path_depth(link), self._classify_section(link, url, soup, root_domain), ""))
                else:
                    if link not in external_links and link not in self._visited:
                        external_links[link] = section

    def _classify_section(self, child_url: str, parent_url: str, parent_soup: BeautifulSoup, root_domain: str) -> str:
        first_seg = _first_path_segment(child_url)
        if first_seg and first_seg in _SECTION_KEYWORDS:
            return _SECTION_KEYWORDS[first_seg]
        inferred = _infer_section_from_url(child_url)
        if inferred != "general":
            return inferred
        for a_tag in parent_soup.find_all("a", href=True):
            try:
                href = urljoin(parent_url, a_tag["href"]).split("#")[0]
            except Exception:
                continue
            if href == child_url:
                inferred = _infer_section_from_url(child_url, a_tag.get_text(strip=True))
                if inferred != "general":
                    return inferred
        parent_seg = _first_path_segment(parent_url)
        if parent_seg and parent_seg in _SECTION_KEYWORDS:
            return _SECTION_KEYWORDS[parent_seg]
        return "general"

    # ------------------------------------------------------------------
    # Page scrapers
    # ------------------------------------------------------------------

    def _scrape_internal_page(self, url: str, section: str) -> Optional[ScrapedDocument]:
        soup = self._fetch_soup(url)
        if soup is None:
            return None
        content = _extract_plain_text(soup)
        return self._make_doc(title=_page_title(soup, url), section=section, url=url, content=content, doc_type="text") if content else None

    def _scrape_external_page(self, url: str, section: str) -> Optional[ScrapedDocument]:
        # Cache hit path
        if self.cache:
            cached = self.cache.get_page(url)
            if cached:
                html = cached["html"]
                final_url = cached.get("final_url", url)
                ct = cached.get("content_type", "")
                if _is_pdf_ct(ct) or _is_office_ct(ct):
                    content = BeautifulSoup(html, "html.parser").get_text()
                else:
                    content = _extract_article_text(html, final_url)
                    if not content:
                        content = _extract_plain_text(BeautifulSoup(html, "html.parser"))
                if not content:
                    return None
                soup = BeautifulSoup(html, "html.parser")
                return self._make_doc(title=_page_title(soup, url), section=section, url=url, content=content, doc_type="text")

        # Live fetch
        resp = self._do_get(url)
        if resp is None:
            return None

        final_url = resp.url or url
        ct = resp.headers.get("Content-Type", "")

        # Document / binary files (Gemini-first)
        if (_is_pdf_ct(ct) or _is_pdf_url(final_url) or _is_pdf_url(url)
                or _is_office_ct(ct) or _is_office_url(final_url) or _is_office_url(url)):
            logger.info("  [Gemini file extraction] treating as document: %s", final_url)
            filename = final_url.split("/")[-1].split("?")[0] if "/" in final_url else None
            # Use the same Gemini-backed file extractor for PDFs & office files
            extracted = self._file_stage3_gemini(resp.content, final_url, filename=filename, mime_hint=ct)
            if not extracted:
                logger.warning("Document extraction returned empty for %s", final_url)
                return None
            wrapper = f"<html><head><title>FILE: {final_url}</title></head><body><pre>{extracted}</pre></body></html>"
            if self.cache:
                self.cache.set_page(url, final_url, wrapper, ct or "application/octet-stream")
                if final_url != url:
                    self.cache.set_page(final_url, final_url, wrapper, ct or "application/octet-stream")
            self._visited.add(final_url)
            return self._make_doc(title=f"File: {filename or final_url.split('/')[-1]}", section=section, url=url, content=extracted, doc_type="text")

        # HTML path
        html = resp.text
        if self.cache:
            self.cache.set_page(url, final_url, html, ct or "text/html")
            if final_url != url:
                self.cache.set_page(final_url, final_url, html, ct or "text/html")
        self._visited.add(final_url)

        content = _extract_article_text(html, final_url)
        if not content:
            content = _extract_plain_text(BeautifulSoup(html, "html.parser"))
        if not content:
            return None
        soup = BeautifulSoup(html, "html.parser")
        return self._make_doc(title=_page_title(soup, url), section=section, url=url, content=content, doc_type="text")

    # ------------------------------------------------------------------
    # Gemini file extraction (upload -> generate_content -> delete)
    # ------------------------------------------------------------------

    def _file_stage3_gemini(self, file_bytes: bytes, source_url: str, filename: Optional[str] = None, mime_hint: Optional[str] = None) -> str:
        """
        Upload a single file to Gemini Files API, ask the model to extract content
        (spreadsheet -> CSV/markdown per sheet; doc -> full text; slides -> slides),
        then delete the uploaded file (best-effort). Does NOT list files.
        """
        if not self._gemini:
            logger.warning("Gemini client not configured; cannot extract file: %s", source_url)
            return ""

        tmp_path: Optional[str] = None
        uploaded = None
        try:
            # determine suffix from filename or source_url
            suffix = ""
            if filename and "." in filename:
                suffix = "." + filename.split(".")[-1]
            elif source_url and "." in source_url.split("/")[-1]:
                suffix = "." + source_url.split("/")[-1].split("?")[0].split(".")[-1]
                if suffix and not suffix.startswith("."):
                    suffix = "." + suffix

            fd, tmp_path = tempfile.mkstemp(suffix=suffix or "")
            os.close(fd)
            with open(tmp_path, "wb") as fh:
                fh.write(file_bytes)

            logger.info("  [Gemini file] uploading %s (%d KB)", source_url, len(file_bytes) // 1024)
            uploaded = self._gemini.files.upload(file=tmp_path)

            # Choose prompt and mime_type hint based on extension/mime
            ext = (filename or source_url or "").lower().split("?")[0]
            prompt = (
                "Extract the file's textual content in a plain, machine-friendly form. "
                "If it's a spreadsheet, produce a CSV or markdown table per sheet. "
                "If it's a document, preserve headings, paragraphs, lists and tables. "
                "If it's a presentation, list slide-by-slide content. "
                "Return the raw extracted content only (no commentary)."
            )
            mime_type = mime_hint or ""

            if ext.endswith(".pdf") or (mime_hint and "pdf" in mime_hint):
                prompt = (
                    "Extract ALL text from this PDF verbatim. Preserve headings, paragraphs, tables and bullet lists. "
                    "Use LaTeX for mathematical formulas. Do not summarise or omit any content."
                )
                mime_type = "application/pdf"
            elif ext.endswith((".xlsx", ".xls", ".xlsm", ".xlsb", ".csv")) or (mime_hint and "spreadsheet" in mime_hint) or ext.endswith(".ods"):
                prompt = (
                    "Extract the spreadsheet contents. For each sheet, produce a CSV block or a markdown table with the sheet name as a header. "
                    "Preserve column headers, cell text and numbers. Indicate empty cells as empty. Return only the data and sheet names."
                )
            elif ext.endswith((".docx", ".doc", ".odt")) or (mime_hint and "word" in mime_hint):
                prompt = (
                    "Extract the full text from this document. Preserve headings, paragraphs, lists and tables. "
                    "Return plain text with headings clearly marked. Do not summarise."
                )
            elif ext.endswith((".pptx", ".ppt")) or (mime_hint and "presentation" in mime_hint):
                prompt = (
                    "Extract slide-by-slide content. For each slide, list the slide number, title (if any) and bullet points / text blocks. "
                    "Return in plain text or markdown with slide separators."
                )

            text = ""
            try:
                contents = types.Content(parts=[
                    types.Part(file_data=types.FileData(
                        file_uri=getattr(uploaded, "uri", getattr(uploaded, "name", None)),
                        mime_type=mime_type or None
                    )),
                    types.Part(text=prompt),
                ])
                response = self._gemini.models.generate_content(
                    model=self.gemini_model,
                    contents=contents,
                )
                text = getattr(response, "text", "") or ""
            except Exception as exc:
                logger.debug("generate_content(parts=...) failed; trying fallback form: %s", exc)
                try:
                    response = self._gemini.models.generate_content(
                        model=self.gemini_model,
                        contents=[uploaded, prompt],
                    )
                    text = getattr(response, "text", "") or ""
                except Exception as exc2:
                    logger.error("Gemini generate_content failed for %s: %s", source_url, exc2)
                    text = ""

            # Attempt to delete uploaded file from Gemini storage (best-effort)
            try:
                if getattr(uploaded, "name", None) and hasattr(self._gemini.files, "delete"):
                    self._gemini.files.delete(name=uploaded.name)
            except Exception:
                pass

            return _clean_text(text)

        except Exception as exc:
            logger.error("Gemini file extraction failed [%s]: %s", source_url, exc)
            return ""
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass

    # ------------------------------------------------------------------
    # Video summarisation (cache-aware)
    # ------------------------------------------------------------------

    def _summarise_video(self, yt_url: str, section: str) -> list[ScrapedDocument]:
        if self.cache:
            cached = self.cache.get_video(yt_url)
            if cached is not None:
                logger.info("  [video cache HIT]  %s", yt_url)
                return [self._make_doc(title=self._extract_video_title(cached, yt_url), section=section, url=yt_url, content=cached, doc_type="video_summary")]

        if not self._gemini:
            logger.warning("Gemini client not configured; skipping video: %s", yt_url)
            return []

        logger.info("  [video Gemini call] %s", yt_url)
        prompt = (
            "Please summarise the video thoroughly and provide a full transcript.\n\n"
            "## Summarisation\n<detailed summary here>\n\n"
            "## Transcript\n<full transcript here>"
        )
        try:
            response = self._gemini.models.generate_content(
                model=self.gemini_model,
                contents=types.Content(parts=[
                    types.Part(file_data=types.FileData(file_uri=yt_url)),
                    types.Part(text=prompt),
                ])
            )
            content = response.text or ""
            if self.cache:
                self.cache.set_video(yt_url, content)
            return [self._make_doc(title=self._extract_video_title(content, yt_url), section=section, url=yt_url, content=content, doc_type="video_summary")]
        except Exception as exc:
            logger.error("  Gemini video failed [%s]: %s", yt_url, exc)
            return []

    # ------------------------------------------------------------------
    # HTTP fetch layer
    # ------------------------------------------------------------------

    def _do_get(self, url: str, max_retries: int = 3) -> Optional[requests.Response]:
        for attempt in range(1, max_retries + 1):
            try:
                try:
                    resp = self._session.get(url, timeout=self.timeout, allow_redirects=True, verify=True)
                except requests.exceptions.SSLError:
                    logger.warning("SSL error %s — retrying verify=False", url)
                    resp = self._session.get(url, timeout=self.timeout, allow_redirects=True, verify=False)
                resp.raise_for_status()
                return resp
            except requests.RequestException as exc:
                logger.warning("Fetch attempt %d/%d failed [%s]: %s", attempt, max_retries, url, exc)
                if attempt == max_retries:
                    return None
                time.sleep(0.8 * (2 ** (attempt - 1)) + random.uniform(0, 0.4))
            except Exception as exc:
                logger.warning("Unexpected error fetching %s: %s", url, exc)
                return None
        return None

    def _fetch_soup_and_final(self, url: str) -> tuple[Optional[BeautifulSoup], str]:
        final_url = url
        current_url = url

        for _hop in range(2):  # allow one meta-refresh hop
            resp = self._do_get(current_url)
            if resp is None:
                return None, final_url

            final_url = resp.url or current_url
            ct = resp.headers.get("Content-Type", "").lower()

            # Binary/document files: use Gemini-first extraction
            if (_is_pdf_ct(ct) or _is_pdf_url(final_url) or _is_pdf_url(current_url)
                    or _is_office_ct(ct) or _is_office_url(final_url) or _is_office_url(current_url)):
                logger.info("  [Gemini file] detected document URL/content-type: %s (%s)", final_url, ct)
                try:
                    filename = final_url.split("/")[-1].split("?")[0]
                    extracted = self._file_stage3_gemini(resp.content, final_url, filename=filename, mime_hint=ct)
                    if not extracted:
                        logger.warning("Gemini returned empty for %s", final_url)
                        return None, final_url
                    wrapper = f"<html><head><title>FILE: {final_url}</title></head><body><pre>{extracted}</pre></body></html>"
                    if self.cache:
                        self.cache.set_page(url, final_url, wrapper, ct or "application/octet-stream")
                        if final_url != url:
                            self.cache.set_page(final_url, final_url, wrapper, ct or "application/octet-stream")
                    self._visited.add(final_url)
                    return BeautifulSoup(wrapper, "html.parser"), final_url
                except Exception as exc:
                    logger.warning("Gemini file extraction failed for %s: %s", final_url, exc)
                    return None, final_url

            # HTML path
            html = resp.text
            soup = BeautifulSoup(html, "html.parser")

            # meta-refresh once
            meta = soup.find("meta", attrs={"http-equiv": re.compile(r"refresh", re.I)})
            if meta and meta.get("content"):
                m = re.search(r"url=(.+)", meta["content"], flags=re.I)
                if m and _hop == 0:
                    redirect_target = urljoin(final_url, m.group(1).strip(" '\""))
                    logger.info("Meta-refresh %s -> %s", final_url, redirect_target)
                    current_url = redirect_target
                    continue

            if self.cache:
                self.cache.set_page(url, final_url, html, ct or "text/html")
                if final_url != url:
                    self.cache.set_page(final_url, final_url, html, ct or "text/html")
            self._visited.add(final_url)
            return soup, final_url

        return None, final_url

    def _fetch_soup(self, url: str) -> Optional[BeautifulSoup]:
        if self.cache:
            cached = self.cache.get_page(url)
            if cached:
                return BeautifulSoup(cached["html"], "html.parser")
        soup, _ = self._fetch_soup_and_final(url)
        return soup

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _make_doc(self, title: str, section: str, url: str, content: str, doc_type: str, extra: Optional[dict] = None) -> ScrapedDocument:
        self._doc_counter += 1
        return ScrapedDocument(index=self._doc_counter, title=title, section=section, url=url, content=content, doc_type=doc_type, extra=extra or {})

    def _make_index_doc(self, soup: BeautifulSoup, url: str, section: str, title: str) -> ScrapedDocument:
        """Index doc contains text + links (useful for site index pages)."""
        return self._make_doc(title=title, section=section, url=url, content=_extract_text_with_links(soup, url), doc_type="index")

    @staticmethod
    def _extract_video_title(content: str, fallback_url: str) -> str:
        for line in content.splitlines():
            line = line.strip().lstrip("#").strip()
            if line and len(line) < 200:
                return line
        return f"Video – {fallback_url}"

    def _reset_session(self) -> None:
        self._doc_counter = 0
        self._visited = set()

