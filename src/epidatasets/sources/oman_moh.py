"""
Oman Ministry of Health Annual Health Reports Accessor

Provides access to health statistics from the Sultanate of Oman Ministry of
Health Annual Health Reports, including:
- Morbidity and mortality data (Chapter 9)
- Health status indicators (Chapter 2)
- Health service utilization (Chapter 7)
- Human resources for health (Chapter 4)
- Subnational (governorate) health statistics

Data Sources:
- MOH Statistics: https://moh.gov.om/en/statistics/annual-health-reports/
- GHDX Catalog: https://ghdx.healthdata.org/series/oman-annual-health-report
- Ministry of Health: https://moh.gov.om/

Update Frequency: Annual
Historical Coverage: 1983/1984 to present

License: Public Government Data (Oman)

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


class OmanMOHAccessor(BaseAccessor):
    """
    Accessor for Oman Ministry of Health Annual Health Reports.

    Provides access to:
    - Annual Health Reports (1983/1984–present)
    - Morbidity and mortality statistics
    - Health status indicators
    - Health service utilization data
    - Governorate-level health statistics
    - Other statistical reports

    Example:
        >>> moh = OmanMOHAccessor()
        >>>
        >>> # Get a specific annual report
        >>> report = moh.get_annual_report(year=2023)
        >>>
        >>> # List available reports
        >>> reports = moh.list_available_reports()
        >>>
        >>> # Extract morbidity/mortality data
        >>> df = moh.extract_morbidity_mortality(report["pdf_path"])
        >>>
        >>> # Get health indicators
        >>> indicators = moh.extract_health_indicators(report["pdf_path"])

    Data Sources:
        - MOH Portal: https://moh.gov.om/en/statistics/annual-health-reports/
        - GHDX: https://ghdx.healthdata.org/series/oman-annual-health-report
    """

    source_name: ClassVar[str] = "oman_moh"
    source_description: ClassVar[str] = (
        "Health statistics from Oman Ministry of Health Annual Health Reports, "
        "including morbidity, mortality, health indicators, utilization, "
        "and governorate-level data."
    )
    source_url: ClassVar[str] = (
        "https://moh.gov.om/en/statistics/annual-health-reports/"
    )

    BASE_URL = "https://moh.gov.om"
    REPORTS_URL = "https://moh.gov.om/en/statistics/annual-health-reports/"
    GHDX_URL = "https://ghdx.healthdata.org/series/oman-annual-health-report"

    # Report series metadata from GHDX and MOH
    REPORT_SERIES = {
        "title": "Oman Annual Health Report",
        "previous_title": "Annual Statistical Report",
        "coverage_start": 1983,
        "coverage_end": 2024,
        "geographic_coverage": "National + Governorate",
        "languages": ["English", "Arabic"],
    }

    # Oman governorates
    GOVERNORATES = [
        "Ad Dakhiliyah",
        "Ad Dhahirah",
        "Al Batinah North",
        "Al Batinah South",
        "Al Buraimi",
        "Al Wusta",
        "Ash Sharqiyah North",
        "Ash Sharqiyah South",
        "Dhofar",
        "Masqat",
        "Musandam",
    ]

    # Common diseases and conditions tracked
    COMMUNICABLE_DISEASES = [
        "Malaria",
        "Tuberculosis",
        "Hepatitis",
        "Dengue",
        "Crimean-Congo Hemorrhagic Fever",
        "Poliomyelitis",
        "Measles",
        "Influenza",
        "Typhoid",
        "Leishmaniasis",
    ]

    NON_COMMUNICABLE_DISEASES = [
        "Circulatory diseases",
        "Diabetes mellitus",
        "Malignant neoplasms",
        "Respiratory diseases",
        "Injuries",
    ]

    def __init__(self, cache_dir: Optional[Union[str, Path]] = None):
        """
        Initialize Oman MOH accessor.

        Args:
            cache_dir: Directory to cache downloaded PDFs.
                       Defaults to ~/.cache/epi_data/oman_moh.
        """
        super().__init__()
        if cache_dir:
            self.cache_dir = Path(cache_dir)
        else:
            self.cache_dir = Path.home() / ".cache" / "epi_data" / "oman_moh"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache_ttl = timedelta(days=30)  # Annual reports change rarely
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
        List countries covered by Oman MOH data.

        Returns:
            DataFrame with country information.
        """
        return pd.DataFrame(
            [{"country_code": "OM", "country_name": "Oman"}]
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
        """
        Download a PDF from URL and save to cache.

        Args:
            url: URL to download.
            cache_path: Local path to save the file.

        Returns:
            Path to the downloaded (or cached) PDF.
        """
        if self._is_cache_valid(cache_path):
            logger.info(f"Using cached PDF: {cache_path}")
            return cache_path

        logger.info(f"Downloading PDF from: {url}")
        response = self._session.get(url, timeout=120, stream=True)
        response.raise_for_status()

        content_type = response.headers.get("content-type", "").lower()
        if "pdf" not in content_type and not response.content.startswith(b"%PDF"):
            raise ValueError(f"Expected PDF, got content-type: {content_type}")

        cache_path.write_bytes(response.content)
        logger.info(f"Saved PDF to cache: {cache_path}")
        return cache_path

    def _extract_pdf_text(self, pdf_path: Path) -> str:
        """
        Extract text from a PDF file.

        Args:
            pdf_path: Path to the PDF file.

        Returns:
            Extracted text content.
        """
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
        """
        Attempt to extract tables from a PDF.

        Args:
            pdf_path: Path to the PDF file.

        Returns:
            List of DataFrames extracted from the PDF.
        """
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

        # Fallback text-based heuristic
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

    def _build_annual_report_url(self, year: int) -> str:
        """
        Build a likely URL for an Annual Health Report PDF.

        Based on observed patterns:
        - https://moh.gov.om/media/.../annual-health-report-{YYYY}.pdf
        - GHDX references specific media URLs

        Args:
            year: Report year.

        Returns:
            Best-guess URL for the report PDF.
        """
        # Try the most common patterns
        candidates = [
            f"{self.BASE_URL}/media/annual-health-report-{year}.pdf",
            f"{self.BASE_URL}/media/j50gq14m/annual-health-report-{year}.pdf",
            f"{self.BASE_URL}/en/statistics/annual-health-reports/annual-health-report-{year}/",
        ]
        return candidates[0]

    def list_available_reports(self) -> pd.DataFrame:
        """
        List known Annual Health Reports with their expected URLs.

        Since the MOH website does not always expose a predictable listing,
        this returns the known historical range (1983–2024) with constructed
        URLs that can be probed.

        Returns:
            DataFrame with columns: year, title, expected_url, notes.
        """
        start = self.REPORT_SERIES["coverage_start"]
        end = self.REPORT_SERIES["coverage_end"]

        results = []
        for year in range(start, end + 1):
            title = f"Oman Annual Health Report {year}"
            notes = None
            if year < 2000:
                notes = "May be titled 'Annual Statistical Report'"
            expected_url = self._build_annual_report_url(year)
            results.append(
                {
                    "year": year,
                    "title": title,
                    "expected_url": expected_url,
                    "notes": notes,
                }
            )

        return pd.DataFrame(results)

    def get_annual_report(
        self,
        year: int,
        extract_text: bool = False,
        extract_tables: bool = False,
    ) -> Dict:
        """
        Download and optionally parse an Annual Health Report.

        Args:
            year: Report year.
            extract_text: Whether to extract text from the PDF.
            extract_tables: Whether to extract tables from the PDF.

        Returns:
            Dictionary with:
                - 'year', 'title', 'url', 'pdf_path'
                - 'text' (if extract_text=True)
                - 'tables' (if extract_tables=True)
                - 'error' if download failed.
        """
        url = self._build_annual_report_url(year)
        cache_filename = f"oman_annual_report_{year}.pdf"
        cache_path = self._get_cache_path(cache_filename)

        result = {
            "year": year,
            "title": f"Oman Annual Health Report {year}",
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

    def get_report_range(
        self,
        start_year: int,
        end_year: int,
        extract_text: bool = False,
        extract_tables: bool = False,
    ) -> pd.DataFrame:
        """
        Download a range of Annual Health Reports.

        Args:
            start_year: Starting year.
            end_year: Ending year.
            extract_text: Whether to extract text from PDFs.
            extract_tables: Whether to extract tables from PDFs.

        Returns:
            DataFrame with one row per report attempt.
        """
        results = []
        for year in range(start_year, end_year + 1):
            report = self.get_annual_report(
                year, extract_text=extract_text, extract_tables=extract_tables
            )
            results.append(report)
        return pd.DataFrame(results)

    def extract_morbidity_mortality(self, pdf_path: Union[str, Path]) -> pd.DataFrame:
        """
        Extract morbidity and mortality data from a report PDF.

        Targets Chapter 9 content (Morbidity and Mortality) by searching
        for disease names and associated numeric values in the extracted text.

        Args:
            pdf_path: Path to the Annual Health Report PDF.

        Returns:
            DataFrame with parsed disease statistics.
        """
        pdf_path = Path(pdf_path)
        text = self._extract_pdf_text(pdf_path)
        if not text:
            return pd.DataFrame()

        rows = []
        lines = text.splitlines()
        all_diseases = self.COMMUNICABLE_DISEASES + self.NON_COMMUNICABLE_DISEASES

        for line in lines:
            for disease in all_diseases:
                # Match disease name and look for nearby numbers
                if disease.lower() in line.lower():
                    numbers = re.findall(r"\b(\d+(?:,\d{3})*)\b", line)
                    if numbers:
                        row = {"disease": disease, "raw_line": line}
                        for i, num in enumerate(numbers[:4]):
                            row[f"value_{i + 1}"] = int(num.replace(",", ""))
                        rows.append(row)
                    break

        return pd.DataFrame(rows)

    def extract_health_indicators(self, pdf_path: Union[str, Path]) -> pd.DataFrame:
        """
        Extract health status indicators from a report PDF.

        Targets Chapter 2 content (Health Status Indicators) such as
        life expectancy, mortality rates, and hospital indicators.

        Args:
            pdf_path: Path to the Annual Health Report PDF.

        Returns:
            DataFrame with parsed health indicators.
        """
        pdf_path = Path(pdf_path)
        text = self._extract_pdf_text(pdf_path)
        if not text:
            return pd.DataFrame()

        indicators = []
        indicator_patterns = [
            (r"life expectancy.*?(\d+\.?\d*)", "life_expectancy"),
            (r"infant mortality.*?(\d+\.?\d*)", "infant_mortality_rate"),
            (r"maternal mortality.*?(\d+\.?\d*)", "maternal_mortality_ratio"),
            (r"under.?five mortality.*?(\d+\.?\d*)", "under5_mortality_rate"),
            (r"crude death rate.*?(\d+\.?\d*)", "crude_death_rate"),
            (r"crude birth rate.*?(\d+\.?\d*)", "crude_birth_rate"),
            (r"total fertility rate.*?(\d+\.?\d*)", "total_fertility_rate"),
            (r"hospital beds.*?(\d+(?:,\d{3})*)", "hospital_beds"),
            (r"physicians.*?(\d+(?:,\d{3})*)", "physicians"),
            (r"nurses.*?(\d+(?:,\d{3})*)", "nurses"),
        ]

        for pattern, indicator_name in indicator_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE | re.DOTALL)
            for match in matches:
                value_str = match.group(1).replace(",", "")
                try:
                    value = float(value_str)
                    indicators.append(
                        {
                            "indicator": indicator_name,
                            "value": value,
                            "raw_text": match.group(0)[:200],
                        }
                    )
                except ValueError:
                    continue

        return pd.DataFrame(indicators)

    def extract_health_utilization(self, pdf_path: Union[str, Path]) -> pd.DataFrame:
        """
        Extract health service utilization data from a report PDF.

        Targets Chapter 7 content (Utilization of Health Services).

        Args:
            pdf_path: Path to the Annual Health Report PDF.

        Returns:
            DataFrame with parsed utilization statistics.
        """
        pdf_path = Path(pdf_path)
        text = self._extract_pdf_text(pdf_path)
        if not text:
            return pd.DataFrame()

        utilization_patterns = [
            (r"outpatient visits.*?(\d+(?:,\d{3})*)", "outpatient_visits"),
            (r"inpatient admissions.*?(\d+(?:,\d{3})*)", "inpatient_admissions"),
            (r"surgeries.*?(\d+(?:,\d{3})*)", "surgeries"),
            (r"deliveries.*?(\d+(?:,\d{3})*)", "deliveries"),
            (r"emergency visits.*?(\d+(?:,\d{3})*)", "emergency_visits"),
            (r"dental visits.*?(\d+(?:,\d{3})*)", "dental_visits"),
        ]

        rows = []
        for pattern, metric_name in utilization_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE | re.DOTALL)
            for match in matches:
                value_str = match.group(1).replace(",", "")
                try:
                    value = float(value_str)
                    rows.append(
                        {
                            "metric": metric_name,
                            "value": value,
                            "raw_text": match.group(0)[:200],
                        }
                    )
                except ValueError:
                    continue

        return pd.DataFrame(rows)

    def get_governorate_data(self, pdf_path: Union[str, Path]) -> pd.DataFrame:
        """
        Extract governorate-level data from a report PDF.

        Searches for tables or text blocks that mention governorates
        alongside numeric health indicators.

        Args:
            pdf_path: Path to the Annual Health Report PDF.

        Returns:
            DataFrame with governorate-level statistics.
        """
        pdf_path = Path(pdf_path)
        text = self._extract_pdf_text(pdf_path)
        if not text:
            return pd.DataFrame()

        rows = []
        lines = text.splitlines()

        for line in lines:
            for gov in self.GOVERNORATES:
                if gov.lower() in line.lower():
                    numbers = re.findall(r"\b(\d+(?:,\d{3})*)\b", line)
                    if numbers:
                        row = {"governorate": gov, "raw_line": line}
                        for i, num in enumerate(numbers[:4]):
                            row[f"value_{i + 1}"] = int(num.replace(",", ""))
                        rows.append(row)
                    break

        return pd.DataFrame(rows)

    def get_other_statistical_reports(self) -> pd.DataFrame:
        """
        Scrape the MOH statistics page for other available reports.

        Returns:
            DataFrame with columns: title, url, category.
        """
        try:
            response = self._session.get(
                f"{self.BASE_URL}/en/statistics/", timeout=30
            )
            response.raise_for_status()
        except requests.RequestException as e:
            logger.error(f"Failed to fetch statistics page: {e}")
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
                for kw in ["report", "statistical", "governorate", "health"]
            ):
                full_url = urljoin(self.BASE_URL, href)
                category = None
                if "governorate" in lower_text:
                    category = "governorate"
                elif "annual" in lower_text:
                    category = "annual"
                elif "statistical" in lower_text:
                    category = "statistical"

                results.append(
                    {"title": text, "url": full_url, "category": category}
                )

        return pd.DataFrame(results)
