"""
Professional Table Renderer
============================
- Wraps <table> HTML with optional <h2> title and <p> footnote.
- Injects CSS from StyleEngine into <head>.
- Uses Playwright to screenshot exact table bounds.
- Applies augmentation: Clean / Medium / Heavy noise pipeline.

Noise Distribution: 30% Clean, 40% Medium, 30% Heavy
"""

import asyncio
import os
import random
from typing import Optional

import cv2
import numpy as np
from playwright.async_api import async_playwright, Browser, BrowserContext


# ---------------------------------------------------------------------------
# Optional title / footnote pools
# ---------------------------------------------------------------------------

TITLES = [
    "Bill of Materials — Revision {rev}",
    "Financial Summary — {period}",
    "Component Inventory Report",
    "Quarterly Performance Overview",
    "Technical Specification Matrix",
    "Cost Analysis by Product Line",
    "Hierarchical Parts Breakdown",
    "Statistical Summary by Region",
    "Annual Budget Allocation Table",
    "Key Performance Indicators Dashboard",
]

FOOTNOTES = [
    "All figures in USD unless otherwise stated.",
    "Proprietary data — authorized access only.",
    "Source: Internal ERP system as of Q{q} {year}.",
    "Values rounded to two decimal places.",
    "N/A indicates data not available for the reporting period.",
    "Totals may not sum due to rounding.",
    "Preliminary figures subject to audit adjustment.",
]


def _random_title() -> Optional[str]:
    if random.random() < 0.55:
        tmpl = random.choice(TITLES)
        return tmpl.format(
            rev=random.choice(["A", "B", "C", "D"]),
            period=random.choice(["FY2023", "FY2024", "H1 2024", "Q3 2024"]),
        )
    return None


def _random_footnote() -> Optional[str]:
    if random.random() < 0.40:
        tmpl = random.choice(FOOTNOTES)
        return tmpl.format(
            q=random.randint(1, 4),
            year=random.choice([2023, 2024, 2025]),
        )
    return None


# ---------------------------------------------------------------------------
# HTML Scaffolding
# ---------------------------------------------------------------------------

def build_full_html(table_html: str, css: str,
                    title: Optional[str] = None,
                    footnote: Optional[str] = None) -> str:
    title_tag = f'<h2>{title}</h2>' if title else ""
    note_tag = (
        f'<p class="footnote">* {footnote}</p>' if footnote else ""
    )
    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
{css}
</style>
</head>
<body>
{title_tag}
{table_html}
{note_tag}
</body>
</html>"""


# ---------------------------------------------------------------------------
# Augmentation helpers
# ---------------------------------------------------------------------------

def _add_gaussian_noise(img: np.ndarray, sigma: float) -> np.ndarray:
    noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
    noisy = img.astype(np.float32) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)


def _add_sensor_grain(img: np.ndarray, intensity: float = 0.03) -> np.ndarray:
    grain = np.random.uniform(-intensity * 255, intensity * 255, img.shape)
    result = img.astype(np.float32) + grain
    return np.clip(result, 0, 255).astype(np.uint8)


def _apply_rotation(img: np.ndarray, max_angle: float = 0.5) -> np.ndarray:
    angle = np.random.uniform(-max_angle, max_angle)
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(
        img, M, (w, h),
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )


def _apply_low_res_scale(img: np.ndarray) -> np.ndarray:
    """Simulate low-resolution scanning by downscaling then upscaling."""
    h, w = img.shape[:2]
    scale = random.uniform(0.55, 0.80)
    small = cv2.resize(img, (max(1, int(w * scale)), max(1, int(h * scale))),
                       interpolation=cv2.INTER_LINEAR)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def apply_augmentation(image_path: str, level: str = "clean") -> None:
    """
    Applies in-place noise augmentation.
    level: "clean" | "medium" | "heavy"
    """
    if level == "clean":
        return

    img = cv2.imread(image_path)
    if img is None:
        return

    if level == "medium":
        img = _add_gaussian_noise(img, sigma=random.uniform(1.0, 3.0))
        img = _add_sensor_grain(img, intensity=random.uniform(0.01, 0.025))

    elif level == "heavy":
        img = _add_gaussian_noise(img, sigma=random.uniform(4.0, 8.0))
        img = _add_sensor_grain(img, intensity=random.uniform(0.03, 0.06))
        img = _apply_rotation(img, max_angle=random.uniform(0.1, 0.5))
        if random.random() < 0.60:
            img = _apply_low_res_scale(img)

    cv2.imwrite(image_path, img)


def sample_noise_level() -> str:
    return random.choices(
        ["clean", "medium", "heavy"],
        weights=[0.30, 0.40, 0.30]
    )[0]


# ---------------------------------------------------------------------------
# Playwright-based renderer
# ---------------------------------------------------------------------------

class TableRenderer:
    """
    Manages a shared Playwright browser for efficient async batch rendering.
    Use as an async context manager.
    """

    def __init__(self, output_dir: str = "data/synthetic/images"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self._playwright = None
        self._browser: Optional[Browser] = None

    async def __aenter__(self) -> "TableRenderer":
        self._playwright = await async_playwright().start()
        self._browser = await self._playwright.chromium.launch(headless=True)
        return self

    async def __aexit__(self, *args) -> None:
        if self._browser:
            await self._browser.close()
        if self._playwright:
            await self._playwright.stop()

    async def render(
        self,
        table_html: str,
        css: str,
        filename: str,
        title: Optional[str] = None,
        footnote: Optional[str] = None,
        noise_level: str = "clean",
    ) -> str:
        """
        Renders table to PNG, applies augmentation, returns absolute image path.
        """
        full_html = build_full_html(table_html, css, title, footnote)
        output_path = os.path.join(self.output_dir, f"{filename}.png")

        context: BrowserContext = await self._browser.new_context(
            viewport={"width": 1280, "height": 960}
        )
        page = await context.new_page()
        await page.set_content(full_html, wait_until="networkidle")

        table_el = await page.query_selector("table")
        if table_el:
            await table_el.screenshot(path=output_path, omit_background=False)
        else:
            await page.screenshot(path=output_path, full_page=True)

        await context.close()

        apply_augmentation(output_path, level=noise_level)
        return output_path