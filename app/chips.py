"""The chip catalogue: the five supported ESP32-family targets.

This is the single source of truth for chip identifiers, the hardware badge
shown by the frontend, and the per-chip sample questions. The frontend reads
all of it from `GET /api/chips`.
"""
from __future__ import annotations

from typing import Final

COMMON: Final = "common"

CHIP_IDS: Final[tuple[str, ...]] = ("esp32", "esp32-s3", "esp32-c3", "esp32-c6", "esp32-s2")
ALL_TAGS: Final[tuple[str, ...]] = CHIP_IDS + (COMMON,)

CHIP_CATALOG: Final[dict[str, dict]] = {
    "esp32": {
        "id": "esp32",
        "label": "ESP32",
        "architecture": "Xtensa LX6",
        "cores": "Dual-core",
        "max_frequency_mhz": 240,
        "wifi": "Wi-Fi 4 (802.11 b/g/n, 2.4 GHz)",
        "bluetooth": "Bluetooth 4.2 BR/EDR + BLE",
        "usb": "No native USB (external USB-UART bridge)",
        "extra_radio": None,
        "gpio_count": 34,
        "sample_questions": [
            "How many ADC channels does the chip have, and which ADC stops working when Wi-Fi is on?",
            "Which GPIO pins are input-only?",
            "Which strapping pin selects the flash voltage, and what happens if it is pulled high?",
            "What is the deep-sleep current with the RTC timer and RTC memory powered?",
            "Which pins are the two DAC outputs?",
        ],
    },
    "esp32-s3": {
        "id": "esp32-s3",
        "label": "ESP32-S3",
        "architecture": "Xtensa LX7",
        "cores": "Dual-core",
        "max_frequency_mhz": 240,
        "wifi": "Wi-Fi 4 (802.11 b/g/n, 2.4 GHz)",
        "bluetooth": "Bluetooth 5 (LE)",
        "usb": "USB OTG + USB Serial/JTAG",
        "extra_radio": None,
        "gpio_count": 45,
        "sample_questions": [
            "Which GPIOs carry USB D+ and D-?",
            "What are the strapping pins?",
            "Which extra GPIOs are taken when the module has octal flash or PSRAM?",
            "What is the deep-sleep current?",
            "At which flash offset is the bootloader written?",
        ],
    },
    "esp32-c3": {
        "id": "esp32-c3",
        "label": "ESP32-C3",
        "architecture": "RISC-V (RV32IMC)",
        "cores": "Single-core",
        "max_frequency_mhz": 160,
        "wifi": "Wi-Fi 4 (802.11 b/g/n, 2.4 GHz)",
        "bluetooth": "Bluetooth 5 (LE)",
        "usb": "USB Serial/JTAG (no OTG)",
        "extra_radio": None,
        "gpio_count": 22,
        "sample_questions": [
            "Which pin do I hold low to enter download mode?",
            "How many ADC channels are available and on which GPIOs?",
            "Which pins are USB D+ and D-?",
            "How many LED PWM channels does it have?",
            "What are the default UART0 TX and RX pins?",
        ],
    },
    "esp32-c6": {
        "id": "esp32-c6",
        "label": "ESP32-C6",
        "architecture": "RISC-V (HP core + LP core)",
        "cores": "Single-core + LP core",
        "max_frequency_mhz": 160,
        "wifi": "Wi-Fi 6 (802.11ax, 2.4 GHz)",
        "bluetooth": "Bluetooth 5 (LE)",
        "usb": "USB Serial/JTAG (no OTG)",
        "extra_radio": "IEEE 802.15.4 (Zigbee / Thread)",
        "gpio_count": 30,
        "sample_questions": [
            "Does the chip support 5 GHz Wi-Fi?",
            "Which protocols can run on the IEEE 802.15.4 radio?",
            "How many ADC channels are there and on which pins?",
            "What is the maximum frequency of the low-power core?",
            "What are the strapping pins?",
        ],
    },
    "esp32-s2": {
        "id": "esp32-s2",
        "label": "ESP32-S2",
        "architecture": "Xtensa LX7",
        "cores": "Single-core",
        "max_frequency_mhz": 240,
        "wifi": "Wi-Fi 4 (802.11 b/g/n, 2.4 GHz)",
        "bluetooth": "No Bluetooth",
        "usb": "USB OTG (full-speed)",
        "extra_radio": None,
        "gpio_count": 43,
        "sample_questions": [
            "Does this chip have Bluetooth?",
            "What is the ADC resolution and how many channels are there?",
            "Which pins are the DAC outputs?",
            "Which strapping pin selects the VDD_SPI voltage?",
            "How many touch sensor pins are available?",
        ],
    },
}


def chip_label(chip_id: str) -> str:
    """Human-readable name for a chip tag (`esp32-s3` -> `ESP32-S3`)."""
    if chip_id == COMMON:
        return "Common (all chips)"
    entry = CHIP_CATALOG.get(chip_id)
    return entry["label"] if entry else chip_id.upper()
