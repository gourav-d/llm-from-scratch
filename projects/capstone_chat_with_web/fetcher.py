"""
fetcher.py -- Phase 1: Fetch a URL and extract clean text.

Pipeline:
  URL --> HTTP GET --> raw HTML --> text extractor --> clean text

Three extractors tried in order (see config.EXTRACTOR_ORDER):
  1. trafilatura  -- smart, removes nav/ads/footers automatically
  2. justext      -- boilerplate removal via stopword lists
  3. naive        -- just strips all HTML tags (last resort)

C# analogy: like HttpClient.GetStringAsync() + HtmlAgilityPack for parsing.
"""

import urllib.request
from html.parser import HTMLParser
from datetime import datetime

import trafilatura
import justext

import config


# ---------------------------------------------------------------------------
# Naive HTML Stripper (same as in notebook 19!)
# ---------------------------------------------------------------------------
# C# analogy: implements an event-based parser (like SAX XML parser in .NET)
class NaiveHTMLStripper(HTMLParser):
    """Strips all HTML tags, keeps only visible text."""

    def __init__(self):
        super().__init__()
        self.text_parts = []
        # Tags whose content we want to skip entirely (not visible to users)
        self.skip_tags = {"script", "style", "noscript", "head"}
        self.currently_skipping = False

    def handle_starttag(self, tag, attrs):
        if tag in self.skip_tags:
            self.currently_skipping = True

    def handle_endtag(self, tag):
        if tag in self.skip_tags:
            self.currently_skipping = False

    def handle_data(self, data):
        if not self.currently_skipping:
            self.text_parts.append(data)

    def get_text(self):
        # Join all parts, then collapse multiple spaces/newlines into one space
        return " ".join("".join(self.text_parts).split())


# ---------------------------------------------------------------------------
# Individual extractors
# ---------------------------------------------------------------------------

def _extract_trafilatura(raw_html: str) -> str | None:
    """Use trafilatura to extract main content."""
    return trafilatura.extract(raw_html)


def _extract_justext(raw_bytes: bytes) -> str | None:
    """Use justext to remove boilerplate, keep content paragraphs."""
    paragraphs = justext.justext(raw_bytes, justext.get_stoplist("English"))
    # is_boilerplate = True means nav/footer/ads -- we skip those
    content = "\n".join(p.text for p in paragraphs if not p.is_boilerplate)
    return content if content.strip() else None


def _extract_naive(raw_html: str) -> str | None:
    """Strip all HTML tags -- last resort."""
    stripper = NaiveHTMLStripper()
    stripper.feed(raw_html)
    text = stripper.get_text()
    return text if text.strip() else None


# ---------------------------------------------------------------------------
# Main fetch function
# ---------------------------------------------------------------------------

def fetch_and_clean(url: str) -> dict | None:
    """
    Fetch a URL and return clean extracted text.

    Returns a dict:
        {
            "url":        "https://...",
            "text":       "Clean extracted text...",
            "extractor":  "trafilatura",   # which extractor worked
            "char_count": 4231,
            "fetched_at": "2026-09-01T10:30:00"
        }

    Returns None if the URL failed or text was too short.

    C# analogy: like a method returning Result<WebPage, Error> in functional style.
    """
    print(f"  Fetching: {url}")

    # --- Step 1: HTTP GET ---
    # We set a User-Agent header to look like a browser.
    # Some sites block requests with no User-Agent (bot detection).
    try:
        request = urllib.request.Request(
            url,
            headers={"User-Agent": "Mozilla/5.0 (compatible; LearningBot/1.0)"}
        )
        with urllib.request.urlopen(request, timeout=config.FETCH_TIMEOUT) as response:
            raw_bytes = response.read()
            # Decode bytes to string -- most web pages are UTF-8
            raw_html = raw_bytes.decode("utf-8", errors="replace")
    except Exception as e:
        print(f"  ERROR fetching {url}: {e}")
        return None

    # --- Step 2: Try extractors in order ---
    # C# analogy: like a chain of responsibility pattern
    extractors = {
        "trafilatura": lambda: _extract_trafilatura(raw_html),
        "justext":     lambda: _extract_justext(raw_bytes),
        "naive":       lambda: _extract_naive(raw_html),
    }

    extracted_text = None
    used_extractor = None

    for name in config.EXTRACTOR_ORDER:
        try:
            text = extractors[name]()
            if text and len(text) >= config.MIN_TEXT_LENGTH:
                extracted_text = text
                used_extractor = name
                break
            else:
                print(f"  {name}: too short or empty, trying next...")
        except Exception as e:
            print(f"  {name}: failed ({e}), trying next...")

    if extracted_text is None:
        print(f"  All extractors failed for {url}")
        return None

    print(f"  OK -- {used_extractor} extracted {len(extracted_text):,} chars")

    return {
        "url":        url,
        "text":       extracted_text,
        "extractor":  used_extractor,
        "char_count": len(extracted_text),
        "fetched_at": datetime.now().isoformat(),
    }


def fetch_multiple(urls: list[str]) -> list[dict]:
    """
    Fetch and clean multiple URLs.

    Returns only successful results (failed URLs are skipped with a warning).
    C# analogy: like urls.Select(FetchAndClean).Where(r => r != null).ToList()
    """
    results = []
    for i, url in enumerate(urls, 1):
        print(f"\n[{i}/{len(urls)}] Processing URL...")
        result = fetch_and_clean(url)
        if result:
            results.append(result)
        else:
            print(f"  Skipped: {url}")

    print(f"\nFetched {len(results)}/{len(urls)} URLs successfully.")
    return results
