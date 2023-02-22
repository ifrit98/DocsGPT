"""Beautiful Soup Web scraper."""
from typing import Any, Callable, Dict, List, Optional, Tuple

from gpt_index.readers.base import BaseReader
from gpt_index.readers.schema.base import Document


def _substack_reader(soup: Any) -> Tuple[str, Dict[str, Any]]:
    """Extract text from Substack blog post."""
    extra_info = {
        "Title of this Substack post": soup.select_one("h1.post-title").getText(),
        "Subtitle": soup.select_one("h3.subtitle").getText(),
        "Author": soup.select_one("span.byline-names").getText(),
    }
    text = soup.select_one("div.available-content").getText()
    return text, extra_info


def _readthedocs_reader(soup: Any) -> Tuple[str, Dict[str, Any]]:
    """Extract text from a ReadTheDocs documentation site"""
    text = soup.find_all("main", {"id": "main-content"})[0].get_text()
    return "\n".join([t for t in text.split("\n") if t]), {}


DEFAULT_WEBSITE_EXTRACTOR: Dict[str, Callable[[Any], Tuple[str, Dict[str, Any]]]] = {
    "substack.com": _substack_reader,
    "readthedocs.io": _readthedocs_reader,
}


class BeautifulSoupWebReader(BaseReader):
    """BeautifulSoup web page reader.

    Reads pages from the web.
    Requires the `bs4` and `urllib` packages.

    Args:
        file_extractor (Optional[Dict[str, Callable]]): A mapping of website
            hostname (e.g. google.com) to a function that specifies how to
            extract text from the BeautifulSoup obj. See DEFAULT_WEBSITE_EXTRACTOR.
    """

    def __init__(
        self,
        website_extractor: Optional[Dict[str, Callable]] = None,
    ) -> None:
        """Initialize with parameters."""
        self.website_extractor = website_extractor or DEFAULT_WEBSITE_EXTRACTOR

    def load_data(
        self, urls: List[str], custom_hostname: Optional[str] = None
    ) -> List[Document]:
        """Load data from the urls.

        Args:
            urls (List[str]): List of URLs to scrape.
            custom_hostname (Optional[str]): Force a certain hostname in the case
                a website is displayed under custom URLs (e.g. Substack blogs)

        Returns:
            List[Document]: List of documents.

        """
        from urllib.parse import urlparse

        import requests
        from bs4 import BeautifulSoup

        documents = []
        for url in urls:
            try:
                page = requests.get(url)
            except Exception:
                raise ValueError(f"One of the inputs is not a valid url: {url}")

            hostname = custom_hostname or urlparse(url).hostname or ""

            soup = BeautifulSoup(page.content, "html.parser")

            data = ""
            extra_info = {"URL": url}
            if hostname in self.website_extractor:
                data, metadata = self.website_extractor[hostname](soup)
                extra_info.update(metadata)
            else:
                data = soup.getText()

            documents.append(Document(data, extra_info=extra_info))

        return documents
