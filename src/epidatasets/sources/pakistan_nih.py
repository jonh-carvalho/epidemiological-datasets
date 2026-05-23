"""
Pakistan NIH IDSR Surveillance Data Accessor

Provides access to disease surveillance data from the Pakistan National Institute
of Health (NIH) Public Health Bulletin, including:
- IDSR Weekly Bulletins (weekly disease surveillance reports)
- Provincial Reports (province/district-level data)
- Seasonal Awareness and Alert Letters (SAAL)
- Advisories

Data Sources:
- PHB Portal: https://phb.nih.org.pk/
- IDSR Weekly: https://phb.nih.org.pk/integratedisease-surveillance-and-response

Update Frequency:
- IDSR Weekly Bulletins: Weekly

License: Public Government Data (Pakistan)

Author: Flávio Codeço Coelho
License: MIT
"""

import logging
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import ClassVar, Dict, List, Optional, Union
from urllib.parse import urljoin

import pandas as pd
import requests
from bs4 import BeautifulSoup

from epidatasets._base import BaseAccessor

logger = logging.getLogger(__name__)


class PakistanNIHAccessor(BaseAccessor):
    """
    Accessor for Pakistan NIH Integrated Disease Surveillance and Response (IDSR) data.

    Provides access to:
    - IDSR Weekly Bulletins (priority disease surveillance)
    - Provincial/district-level reports
    - Seasonal Awareness and Alert Letters (SAAL)
    - Public health advisories

    Example:
        >>> nih = PakistanNIHAccessor()
        >>>
        >>> # Get a specific weekly bulletin
        >>> bulletin = nih.get_weekly_bulletin(year=2025, week=10)
        >>>
        >>> # List available bulletins
        >>> bulletins = nih.list_available_bulletins(2025, 2025)
        >>>
        >>> # Get latest bulletin
        >>> latest = nih.get_latest_bulletin()

    Data Sources:
        - PHB Portal: https://phb.nih.org.pk/
        - IDSR Weekly: https://phb.nih.org.pk/integratedisease-surveillance-and-response
    """

    source_name: ClassVar[str] = "pakistan_nih"
    source_description: ClassVar[str] = (
        "Disease surveillance data from Pakistan NIH Integrated Disease Surveillance "
        "and Response (IDSR), including weekly bulletins, provincial reports, "
        "seasonal alerts, and public health advisories."
    )
    source_url: ClassVar[str] = "https://phb.nih.org.pk/integratedisease-surveillance-and-response"

    BASE_URL = "https://phb.nih.org.pk"
    IDSR_URL = "https://phb.nih.org.pk/integratedisease-surveillance-and-response"
    PROVINCIAL_URL = "https://phb.nih.org.pk/idsr-provincial-reports"
    SAAL_URL = "https://phb.nih.org.pk/saal"
    ADVISORIES_URL = "https://phb.nih.org.pk/advisories"

    # Priority diseases tracked by Pakistan IDSR
    PRIORITY_DISEASES = [
        "COVID-19",
        "Crimean-Congo Hemorrhagic Fever (CCHF)",
        "Dengue Fever",
        "Leishmaniasis",
        "Malaria",
        "Pertussis",
        "Poliomyelitis",
        "Diphtheria",
        "Seasonal Influenza",
        "Typhoid Fever",
        "Measles",
        "Hepatitis",
        "Tuberculosis",
    ]

    # Pakistan provinces
    PROVINCES = [
        "Punjab",
        "Sindh",
        "Khyber Pakhtunkhwa",
        "Balochistan",
        "Gilgit-Baltistan",
        "Azad Jammu and Kashmir",
        "Islamabad Capital Territory",
    ]

    def __init__(self, cache_dir: Optional[Union[str, Path]] = None):
        """
        Initialize Pakistan NIH accessor.

        Args:
            cache_dir: Directory to cache downloaded PDFs.
                       Defaults to ~/.cache/epi_data/pakistan_nih.
        """
        super().__init__()
        if cache_dir:
            self.cache_dir = Path(cache_dir)
        else:
            self.cache_dir = Path.home() / ".cache" / "epi_data" / "pakistan_nih"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache_ttl = timedelta(days=7)
        self._session = requests.Session()
        self._session.headers.update(
            {
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/120.0.0.0 Safari/537.36"
                )
            }
        )

    def list_countries(self) -> pd.DataFrame:
        """
        List countries covered by Pakistan NIH IDSR.

        Returns:
            DataFrame with country information.
        """
        return pd.DataFrame(
            [{"country_code": "PK", "country_name": "Pakistan"}]
        )

    def _get_cache_path(self, filename: str) -> Path:
        """Get full cache path for a file."""
        return self.cache_dir / filename

    def _is_cache_valid(self, cache_path: Path) -> bool:
        """Check if cached file is still valid."""
        if not cache_path.exists():
            return False
        mtime = datetime.fromtimestamp(cache_path.stat().st_mtime)
        return datetime.now() - mtime < self._cache_ttl

    def _download_pdf(self, url: str, cache_path: Path) -> Path:
        """Download a PDF from URL and save to cache."""
        if self._is_cache_valid(cache_path):
            logger.info(f"Using cached PDF: {cache_path}")
            return cache_path

        logger.info(f"Downloading PDF from: {url}")
        response = self._session.get(url, timeout=60, stream=True)
        response.raise_for_status()

        content_type = response.headers.get("content-type", "").lower()
        if "pdf" not in content_type and not response.content.startswith(b"%PDF"):
            raise ValueError(f"Expected PDF, got content-type: {content_type}")

        cache_path.write_bytes(response.content)
        logger.info(f"Saved PDF to cache: {cache_path}")
        return cache_path

    def _extract_pdf_text(self, pdf_path: Path) -> str:
        """Extract text from a PDF file."""
        try:
            from pypdf import PdfReader

            reader = PdfReader(str(pdf_path))
            text_parts = []
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)
            return "\n".join(text_parts)
        except ImportError:
            logger.warning(
                "pypdf not available for PDF text extraction. "
                "Install with: pip install pypdf"
            )
            return ""

    def _extract_pdf_tables(self, pdf_path: Path) -> List[pd.DataFrame]:
        """Attempt to extract tables from a PDF."""
        tables: List[pd.DataFrame] = []

        try:
            import pdfplumber

            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    page_tables = page.extract_tables()
                    for table in page_tables:
                        if table and len(table) > 1:
                            df = pd.DataFrame(table[1:], columns=table[0])
                            tables.append(df)
            return tables
        except ImportError:
            logger.debug("pdfplumber not available for table extraction")

        # Fallback: parse text for table-like structures
        text = self._extract_pdf_text(pdf_path)
        if not text:
            return tables

        lines = text.splitlines()
        table_lines = []
        for line in lines:
            if re.search(r"\d+.*\d+", line) and len(line.split()) >= 3:
                table_lines.append(line.split())

        if table_lines:
            max_cols = max(len(row) for row in table_lines)
            normalized = [row + [""] * (max_cols - len(row)) for row in table_lines]
            if len(normalized) > 1:
                tables.append(pd.DataFrame(normalized[1:], columns=normalized[0]))

        return tables

    def _build_weekly_url(self, year: int, week: int) -> str:
        """Build the URL for an IDSR weekly bulletin."""
        estimated_month = min((week // 4) + 1, 12)
        month_str = f"{estimated_month:02d}"
        return (
            f"https://www.nih.org.pk/wp-content/uploads/"
            f"{year}/{month_str}/Weekly_Report-{week}-{year}.pdf"
        )

    def list_available_bulletins(
        self,
        start_year: int,
        end_year: int,
        max_weeks: int = 52,
    ) -> pd.DataFrame:
        """
        Discover available IDSR weekly bulletins by probing URLs.

        Args:
            start_year: Starting year to check.
            end_year: Ending year to check.
            max_weeks: Maximum week number to check per year.

        Returns:
            DataFrame with columns: year, week, url, status.
        """
        results = []
        current_year = datetime.now().year
        current_week = int(datetime.now().strftime("%U")) + 1

        for year in range(start_year, end_year + 1):
            for week in range(1, max_weeks + 1):
                if year == current_year and week > current_week:
                    continue

                url = self._build_weekly_url(year, week)
                try:
                    response = self._session.head(url, timeout=15, allow_redirects=True)
                    status = "available" if response.status_code == 200 else "missing"
                except requests.RequestException:
                    status = "error"

                results.append(
                    {"year": year, "week": week, "url": url, "status": status}
                )

        return pd.DataFrame(results)

    def get_weekly_bulletin(
        self,
        year: int,
        week: int,
        extract_text: bool = False,
        extract_tables: bool = False,
    ) -> Dict:
        """
        Download and optionally parse an IDSR weekly bulletin.

        Args:
            year: Year of the bulletin.
            week: Epidemiological week number.
            extract_text: Whether to extract text from the PDF.
            extract_tables: Whether to extract tables from the PDF.

        Returns:
            Dictionary with bulletin data or error.
        """
        url = self._build_weekly_url(year, week)
        cache_filename = f"idsr_weekly_{year}_w{week:02d}.pdf"
        cache_path = self._get_cache_path(cache_filename)

        result = {
            "year": year,
            "week": week,
            "url": url,
            "pdf_path": None,
            "text": None,
            "tables": None,
            "error": None,
        }

        try:
            pdf_path = self._download_pdf(url, cache_path)
            result["pdf_path"] = str(pdf_path)

            if extract_text:
                result["text"] = self._extract_pdf_text(pdf_path)
            if extract_tables:
                tables = self._extract_pdf_tables(pdf_path)
                result["tables"] = [df.to_dict(orient="records") for df in tables]

        except requests.HTTPError as e:
            result["error"] = f"HTTP {e.response.status_code}: {e.response.reason}"
        except Exception as e:
            result["error"] = str(e)

        return result

    def get_latest_bulletin(
        self,
        extract_text: bool = False,
        extract_tables: bool = False,
    ) -> Dict:
        """Find and download the most recent available IDSR weekly bulletin."""
        now = datetime.now()
        year = now.year
        week = int(now.strftime("%U")) + 1

        max_attempts = 10
        for _ in range(max_attempts):
            result = self.get_weekly_bulletin(
                year, week, extract_text=extract_text, extract_tables=extract_tables
            )
            if result.get("error") is None:
                return result

            week -= 1
            if week < 1:
                week = 52
                year -= 1

        return {
            "year": None,
            "week": None,
            "url": None,
            "pdf_path": None,
            "text": None,
            "tables": None,
            "error": "No recent bulletin found after 10 attempts",
        }

    def get_bulletin_range(
        self,
        start_year: int,
        start_week: int,
        end_year: int,
        end_week: int,
        extract_text: bool = False,
        extract_tables: bool = False,
    ) -> pd.DataFrame:
        """Download a range of IDSR weekly bulletins."""
        results = []
        year, week = start_year, start_week

        while (year < end_year) or (year == end_year and week <= end_week):
            bulletin = self.get_weekly_bulletin(
                year, week, extract_text=extract_text, extract_tables=extract_tables
            )
            results.append(bulletin)

            week += 1
            if week > 52:
                week = 1
                year += 1

        return pd.DataFrame(results)

    def get_saals(self) -> pd.DataFrame:
        """Scrape the list of Seasonal Awareness and Alert Letters (SAAL)."""
        try:
            response = self._session.get(self.SAAL_URL, timeout=30)
            response.raise_for_status()
        except requests.RequestException as e:
            logger.error(f"Failed to fetch SAAL page: {e}")
            return pd.DataFrame()

        soup = BeautifulSoup(response.text, "html.parser")
        results = []

        for link in soup.find_all("a", href=True):
            href = link["href"]
            text = link.get_text(strip=True)
            if not text:
                continue

            lower_text = text.lower()
            lower_href = href.lower()
            if any(
                kw in lower_text or kw in lower_href
                for kw in ["saal", "seasonal", "alert", "awareness", "letter"]
            ):
                full_url = urljoin(self.BASE_URL, href)
                diseases = [
                    d
                    for d in self.PRIORITY_DISEASES
                    if d.lower().split()[0] in lower_text
                ]
                results.append(
                    {
                        "title": text,
                        "url": full_url,
                        "season": None,
                        "diseases": ", ".join(diseases) if diseases else None,
                    }
                )

        return pd.DataFrame(results)

    def get_advisories(self) -> pd.DataFrame:
        """Scrape the list of public health advisories."""
        try:
            response = self._session.get(self.ADVISORIES_URL, timeout=30)
            response.raise_for_status()
        except requests.RequestException as e:
            logger.error(f"Failed to fetch advisories page: {e}")
            return pd.DataFrame()

        soup = BeautifulSoup(response.text, "html.parser")
        results = []

        for link in soup.find_all("a", href=True):
            href = link["href"]
            text = link.get_text(strip=True)
            if not text:
                continue

            full_url = urljoin(self.BASE_URL, href)
            date_match = re.search(r"(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})", text)
            date_str = date_match.group(1) if date_match else None

            results.append({"title": text, "url": full_url, "date": date_str})

        return pd.DataFrame(results)

    def get_provincial_reports(self, province: Optional[str] = None) -> pd.DataFrame:
        """Scrape the provincial reports page."""
        try:
            response = self._session.get(self.PROVINCIAL_URL, timeout=30)
            response.raise_for_status()
        except requests.RequestException as e:
            logger.error(f"Failed to fetch provincial reports page: {e}")
            return pd.DataFrame()

        soup = BeautifulSoup(response.text, "html.parser")
        results = []

        for link in soup.find_all("a", href=True):
            href = link["href"]
            text = link.get_text(strip=True)
            if not text:
                continue

            full_url = urljoin(self.BASE_URL, href)
            inferred_province = None
            for prov in self.PROVINCES:
                if prov.lower() in text.lower():
                    inferred_province = prov
                    break

            if province and inferred_province != province:
                continue

            results.append(
                {
                    "province": inferred_province,
                    "district": None,
                    "title": text,
                    "url": full_url,
                }
            )

        return pd.DataFrame(results)

    def parse_surveillance_table(self, text: str) -> pd.DataFrame:
        """Parse surveillance data from extracted bulletin text."""
        if not text:
            return pd.DataFrame()

        rows = []
        lines = text.splitlines()

        for line in lines:
            for disease in self.PRIORITY_DISEASES:
                if disease.lower().split()[0] in line.lower():
                    numbers = re.findall(r"\b(\d+)\b", line)
                    if numbers:
                        row = {"disease": disease, "raw_line": line}
                        for i, num in enumerate(numbers[:3]):
                            row[f"value_{i + 1}"] = int(num)
                        rows.append(row)
                    break

        return pd.DataFrame(rows)
