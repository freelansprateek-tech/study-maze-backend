#!/usr/bin/env python
"""Download official Espressif documentation into data/raw/, one folder per chip.

    datasheets   PDF, from espressif.com
    TRMs         PDF, Technical Reference Manuals (only with --with-trm; large)
    ESP-IDF      rendered Programming Guide pages, from docs.espressif.com
    issues       real GitHub issues from espressif/esp-idf (only with --github-issues N)

A manifest (data/raw/manifest.json) records the URL of every file so that
citations link back to the original. Afterwards run:

    python scripts/ingest_docs.py --reset

Examples
    python scripts/fetch_official_docs.py
    python scripts/fetch_official_docs.py --with-trm --github-issues 25
    python scripts/fetch_official_docs.py --chips esp32-c3 esp32-c6 --idf-version v5.3
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from urllib.parse import urljoin

import httpx

BACKEND_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_ROOT))

from app.chips import CHIP_IDS, chip_label  # noqa: E402
from app.config import get_settings  # noqa: E402
from app.ingest.parsers import MANIFEST_NAME  # noqa: E402
from app.ingest.tagger import chip_from_text  # noqa: E402

USER_AGENT = "esp32-docs-fetcher/1.0 (documentation mirror for a local RAG index)"
PDF_BASE = "https://www.espressif.com/sites/default/files/documentation"
IDF_BASE = "https://docs.espressif.com/projects/esp-idf/en"
IDF_TARGET = {"esp32": "esp32", "esp32-s2": "esp32s2", "esp32-s3": "esp32s3", "esp32-c3": "esp32c3", "esp32-c6": "esp32c6"}

# Programming Guide pages that carry chip-specific hardware facts. Pages that do
# not exist for a target (e.g. DAC on the ESP32-C3) return 404 and are skipped.
IDF_PAGES = [
    "get-started/index.html",
    "api-reference/peripherals/gpio.html",
    "api-reference/peripherals/adc_oneshot.html",
    "api-reference/peripherals/adc_continuous.html",
    "api-reference/peripherals/adc_calibration.html",
    "api-reference/peripherals/dac.html",
    "api-reference/peripherals/ledc.html",
    "api-reference/peripherals/i2c.html",
    "api-reference/peripherals/i2s.html",
    "api-reference/peripherals/spi_master.html",
    "api-reference/peripherals/uart.html",
    "api-reference/peripherals/rmt.html",
    "api-reference/peripherals/twai.html",
    "api-reference/peripherals/pcnt.html",
    "api-reference/peripherals/mcpwm.html",
    "api-reference/peripherals/touch_pad.html",
    "api-reference/peripherals/temp_sensor.html",
    "api-reference/peripherals/usb_device.html",
    "api-reference/peripherals/usb_host.html",
    "api-reference/peripherals/sdmmc_host.html",
    "api-reference/system/sleep_modes.html",
    "api-reference/system/power_management.html",
    "api-reference/system/ulp.html",
    "api-reference/system/wdts.html",
    "api-reference/storage/nvs_flash.html",
    "api-guides/bootloader.html",
    "api-guides/partition-tables.html",
    "api-guides/startup.html",
    "api-guides/usb-serial-jtag-console.html",
    "api-guides/usb-otg-console.html",
    "api-guides/flash_psram_config.html",
    "api-guides/wifi.html",
    "api-guides/coexist.html",
    "api-guides/openthread.html",
    "api-guides/low-power-mode/low-power-mode-soc.html",
    "api-guides/tools/idf-monitor.html",
]


_META_REFRESH = re.compile(r'http-equiv="refresh"\s+content="\d+;\s*url=([^"]+)"', re.I)


def download(client: httpx.Client, url: str, destination: Path, expect: str, site_root: str = "") -> str:
    """Fetch `url` to `destination`. Returns the final URL, or "" on any failure
    (in which case nothing is written).

    Pages that have moved are served as a small "meta refresh" stub whose target
    is a path relative to the documentation root; that redirect is followed.
    """
    if destination.exists() and destination.stat().st_size > 3000:
        return url
    for _hop in range(3):
        try:
            response = client.get(url)
        except httpx.HTTPError as exc:
            print(f"    FAILED {url}: {exc}")
            return ""
        if response.status_code != 200:
            print(f"    skip   {response.status_code} {url}")
            return ""
        moved = _META_REFRESH.search(response.text[:2000]) if expect == "html" and len(response.content) < 3000 else None
        if not moved:
            break
        target = moved.group(1)
        # "/x.html" is relative to the documentation root; anything else to this page.
        url = f"{site_root}/{target.lstrip('/')}" if target.startswith("/") else urljoin(url, target)
    content_type = response.headers.get("content-type", "")
    if expect not in content_type:
        print(f"    skip   unexpected content type '{content_type}' {url}")
        return ""
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(response.content)
    print(f"    ok     {len(response.content) / 1024:8.0f} KB  {destination.name}")
    return url


def fetch_github_issues(client: httpx.Client, chip: str, count: int, destination: Path) -> int:
    """Most-discussed closed issues of espressif/esp-idf that name this chip in the title."""
    label = chip_label(chip)
    query = f'repo:espressif/esp-idf is:issue is:closed in:title "{label}"'
    try:
        response = client.get(
            "https://api.github.com/search/issues",
            params={"q": query, "sort": "comments", "order": "desc", "per_page": min(count * 2, 100)},
            headers={"Accept": "application/vnd.github+json"},
        )
    except httpx.HTTPError as exc:
        print(f"    FAILED GitHub search for {label}: {exc}")
        return 0
    if response.status_code != 200:
        print(f"    skip   GitHub search returned {response.status_code} for {label}")
        return 0
    records = []
    for item in response.json().get("items", []):
        body = (item.get("body") or "").strip()
        title = item.get("title") or ""
        # Plain "ESP32" also matches every variant; keep an issue only if its
        # own text is about this chip.
        if len(body.split()) < 30 or chip_from_text(f"{title}\n{title}\n{body}") != chip:
            continue
        records.append({
            "id": f"gh-esp-idf-{item['number']}", "number": item["number"], "chip": chip,
            "title": f"GitHub issue #{item['number']}: {title}", "body": body[:6000],
            "html_url": item["html_url"], "source_type": "github_issue", "section": "Issue report",
        })
        if len(records) >= count:
            break
    if records:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"    ok     {len(records)} GitHub issues for {label}")
    return len(records)


def main() -> int:
    settings = get_settings()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=settings.data_dir / "raw")
    parser.add_argument("--chips", nargs="+", choices=CHIP_IDS, default=list(CHIP_IDS))
    parser.add_argument("--idf-version", default="stable", help="ESP-IDF docs version, e.g. stable, v5.3, latest.")
    parser.add_argument("--with-trm", action="store_true", help="Also download the Technical Reference Manuals.")
    parser.add_argument("--github-issues", type=int, default=0, metavar="N", help="Also fetch N GitHub issues per chip.")
    parser.add_argument("--no-idf", action="store_true", help="Skip the ESP-IDF Programming Guide pages.")
    parser.add_argument("--delay", type=float, default=0.4, help="Pause between requests, in seconds.")
    args = parser.parse_args()

    manifest_path = args.out / MANIFEST_NAME
    manifest: dict[str, dict[str, str]] = {}
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8")).get("files", {})
        except ValueError:
            manifest = {}

    def record(path: Path, url: str, source_type: str, title: str = "") -> None:
        entry = {"url": url, "source_type": source_type}
        if title:
            entry["title"] = title
        manifest[path.relative_to(args.out).as_posix()] = entry

    counts = {"pdf": 0, "idf": 0, "issues": 0}
    with httpx.Client(timeout=120, follow_redirects=True, headers={"User-Agent": USER_AGENT}) as client:
        for chip in args.chips:
            label = chip_label(chip)
            folder = args.out / chip
            print(f"\n{label}")

            name = f"{chip}_datasheet_en.pdf"
            if download(client, f"{PDF_BASE}/{name}", folder / name, "pdf"):
                record(folder / name, f"{PDF_BASE}/{name}", "datasheet", f"{label} Series Datasheet")
                counts["pdf"] += 1
            time.sleep(args.delay)

            if args.with_trm:
                name = f"{chip}_technical_reference_manual_en.pdf"
                if download(client, f"{PDF_BASE}/{name}", folder / name, "pdf"):
                    record(folder / name, f"{PDF_BASE}/{name}", "trm", f"{label} Technical Reference Manual")
                    counts["pdf"] += 1
                time.sleep(args.delay)

            if not args.no_idf:
                base = f"{IDF_BASE}/{args.idf_version}/{IDF_TARGET[chip]}"
                for page in IDF_PAGES:
                    destination = folder / "esp-idf" / page.replace("/", "__")
                    final_url = download(client, f"{base}/{page}", destination, "html", site_root=base)
                    if final_url:
                        record(destination, final_url, "esp-idf-docs")
                        counts["idf"] += 1
                    time.sleep(args.delay)
                    # A guide that was split into a folder redirects to its index
                    # page; fetch the sub-pages that index links to as well.
                    if final_url.endswith("/index.html") and not page.endswith("index.html"):
                        html = destination.read_text(encoding="utf-8", errors="replace")
                        for name in sorted(set(re.findall(r'href="([\w\-]+\.html)(?:#[^"]*)?"', html))):
                            if name == "index.html":
                                continue
                            sub_url = urljoin(final_url, name)
                            sub_path = destination.with_name(f"{destination.stem}__{name}")
                            if download(client, sub_url, sub_path, "html", site_root=base):
                                record(sub_path, sub_url, "esp-idf-docs")
                                counts["idf"] += 1
                            time.sleep(args.delay)

            if args.github_issues > 0:
                destination = folder / "github_issues.json"
                counts["issues"] += fetch_github_issues(client, chip, args.github_issues, destination)
                time.sleep(max(args.delay, 7.0))  # unauthenticated search API: 10 requests per minute

    args.out.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps({"fetched": time.strftime("%Y-%m-%d"), "idf_version": args.idf_version, "files": manifest}, indent=2),
        encoding="utf-8",
    )
    print(f"\nDownloaded {counts['pdf']} PDFs, {counts['idf']} ESP-IDF pages, {counts['issues']} GitHub issues into {args.out}")
    print("Next: python scripts/ingest_docs.py --reset")
    return 0 if counts["pdf"] + counts["idf"] > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
