"""Web scraper for downloading PDF documents.

This script automates the process of scraping a webpage to download PDF files,
ensuring unique filenames, and avoiding duplicates by calculating and comparing
file checksums.
"""

import hashlib
import time
from pathlib import Path
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service

chrome_options = Options()
chrome_options.add_argument("--headless")
chrome_options.add_argument("--no-sandbox")
chrome_options.add_argument("--disable-dev-shm-usage")
chrome_options.add_argument("--disable-gpu")
chrome_options.add_argument("--remote-debugging-port=9222")


def create_folder(directory: Path) -> None:
    """Ensures the folder exists, creating it if necessary.

    Args:
        directory (Path): Path to the folder to check/create.
    """
    if not directory.exists():
        directory.mkdir(parents=True)


def calculate_checksum(file_path: Path) -> str:
    """Calculate the SHA256 checksum of a file.

    Args:
        file_path (Path): Path to the file.

    Returns:
        str: SHA256 checksum of the file.
    """
    sha256_hash = hashlib.sha256()
    with file_path.open("rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

HTTP_OK = 200

def download_pdf(pdf_url: str, file_name: Path) -> Path | None:
    """Downloads PDF from URL and saves it with a unique filename.

    Args:
        pdf_url (str): URL of the PDF to download.
        file_name (Path): Desired file path for the downloaded PDF.

    Returns:
        Path | None: The final file path if download succeeds, else None.
    """
    counter = 1
    while file_name.exists():
        # Append a number to the file name to make it unique
        file_name = file_name.with_stem(f"{file_name.stem}_{counter}")
        counter += 1

    print(f"Downloading {file_name}")
    try:
        response = requests.get(pdf_url, stream=True, timeout=10)
        if response.status_code == HTTP_OK:
            with file_name.open("wb") as f:
                for chunk in response.iter_content(1024):
                    f.write(chunk)
            print(f"Downloaded: {file_name}")
            return file_name
        else:
            print(f"Failed to download: {file_name},"
                  f"status code: {response.status_code}"
            )
            return None
    except Exception as e:
        print(f"Error downloading {file_name}: {e}")
        return None


def get_existing_checksums(download_folder: Path) -> list[str]:
    """Calculate and return the checksums of all existing files in a folder.

    Args:
        download_folder (Path): Path to the folder.

    Returns:
        list[str]: List of SHA256 checksums for existing files.
    """
    checksums = []
    for existing_file in download_folder.iterdir():
        if existing_file.is_file():
            existing_file_checksum = calculate_checksum(existing_file)
            checksums.append(existing_file_checksum)
    return checksums


def is_duplicate(existing_checksums: list[str], new_file_path: Path) -> bool:
    """Checks if the new file's checksum matches any existing checksums.

    Args:
        existing_checksums (list[str]): List of existing file checksums.
        new_file_path (Path): Path to the new file.

    Returns:
        bool: True if the file is a duplicate, False otherwise.
    """
    new_file_checksum = calculate_checksum(new_file_path)
    return new_file_checksum in existing_checksums

two_duplicates = 2 

def main() -> None:
    """Main function to scrape and download PDF files."""
    # Path to the ChromeDriver executable
    chrome_driver_path = Path(
        "~/2024-autumn-building-decarbon/models/scraper/chromedriver"
    ).expanduser()

    # Initialize ChromeDriver session
    service = Service(executable_path=str(chrome_driver_path))
    driver = webdriver.Chrome(service=service, options=chrome_options)

    # Define URLs and folder paths
    base_url = "https://documents.dps.ny.gov/public/MatterManagement/"
    page_url = (
        base_url 
        + "CaseMaster.aspx?MatterCaseNo=22-m-0429&CaseSearch=Search"
    )
    download_folder = Path("data/NY_DPS_Documents")

    # Navigate to the URL and wait for the page to load
    driver.get(page_url)
    time.sleep(20)  # Wait until the page is fully loaded

    # Parse the HTML source code
    page_source = driver.page_source
    driver.quit()

    soup = BeautifulSoup(page_source, "lxml")

    # Create download folder if it doesn't exist
    create_folder(download_folder)

    # Retrieve existing checksums in the folder
    existing_checksums = get_existing_checksums(download_folder)

    # Find the table containing documents
    table = soup.find("table", {"id": "tblPubDoc"})
    if not table:
        print("Document table not found. Verify the HTML structure.")
        return

    rows = table.find("tbody").find_all("tr")
    if not rows:
        print("No rows found in the table.")
        return

    duplicate_count = 0
    for row in rows:
        if duplicate_count >= two_duplicates:
            print("Two consecutive duplicates found, stopping parsing.")
            break

        document_link_tag = row.find("a", href=True)
        if document_link_tag:
            document_title = document_link_tag.text.strip()
            file_name = download_folder / f"{document_title}.pdf"
            pdf_url = urljoin(base_url, document_link_tag["href"])

            downloaded_file_name = download_pdf(pdf_url, file_name)
            if downloaded_file_name:
                if is_duplicate(existing_checksums, downloaded_file_name):
                    print(f"Duplicate found, removing: {downloaded_file_name}")
                    downloaded_file_name.unlink()  # Delete duplicate
                    duplicate_count += 1
                else:
                    duplicate_count = 0
                    existing_checksums.append(
                        calculate_checksum(downloaded_file_name)
                    )
                    print(f"File kept: {downloaded_file_name}")
            time.sleep(0.5)  # Delay to avoid overwhelming the server
        else:
            print("No document link found in this row.")

    print("All viable files have been checked and downloaded.")


if __name__ == "__main__":
    main()
