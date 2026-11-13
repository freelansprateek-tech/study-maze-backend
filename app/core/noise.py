"""Distractor ("noise") documents.

Two sources, chosen with `NOISE_SOURCE`:

    pool  a fixed pool of passages that are unrelated to ESP32 questions. Half
          of them describe *other* microcontrollers, with their own pin counts,
          voltages and memory sizes - the kind of content that tempts a model
          into answering with the wrong chip's numbers.
    llm   passages generated on demand by the generator LLM, cached per run and
          topped up from the pool if generation falls short.

Noise is used in two places: the robustness setting (`noise_ratio`) and the
rearranging module of the indirect-assist branch.
"""
from __future__ import annotations

import hashlib
import random
import re
import threading
from typing import Optional

from app.core.llm_client import LLMClient, LLMError
from app.schemas import ChunkTrace, NoiseOrigin

NOISE_POOL: list[dict[str, str]] = [
    # --- other microcontrollers: hard distractors ---------------------------
    {"title": "ATmega328P Overview",
     "text": "The ATmega328P is an 8-bit AVR microcontroller with 32 KB of flash, 2 KB of SRAM and 1 KB of EEPROM. On the Arduino Uno it runs from a 16 MHz crystal at 5 V. It offers 23 general-purpose I/O lines, a 10-bit ADC with 6 input channels in the DIP package, and three timers that provide 6 PWM outputs."},
    {"title": "Raspberry Pi Pico Hardware Summary",
     "text": "The Raspberry Pi Pico is built around the RP2040, a dual-core Arm Cortex-M0+ running at up to 133 MHz with 264 KB of on-chip SRAM and 2 MB of external QSPI flash. The board exposes 26 multi-function GPIO pins, three of which can be used as 12-bit ADC inputs, and it has 2 UARTs, 2 I2C controllers and 2 SPI controllers."},
    {"title": "STM32F103C8T6 Blue Pill Notes",
     "text": "The STM32F103C8T6 found on Blue Pill boards is an Arm Cortex-M3 microcontroller clocked at up to 72 MHz, with 64 KB of flash and 20 KB of SRAM. It operates from 2.0 V to 3.6 V, provides two 12-bit ADCs with 10 external channels on this package, and includes a CAN interface and a full-speed USB device peripheral."},
    {"title": "nRF52840 Product Brief",
     "text": "The nRF52840 is a multiprotocol SoC based on a 64 MHz Arm Cortex-M4F with 1 MB of flash and 256 KB of RAM. Its 2.4 GHz radio supports Bluetooth 5, Thread and Zigbee, with output power up to +8 dBm. It has 48 GPIOs, a 12-bit ADC with 8 channels, and a full-speed USB 2.0 device controller. It has no Wi-Fi radio."},
    {"title": "ESP8266EX Legacy Notes",
     "text": "The ESP8266EX uses a single-core Tensilica L106 processor at 80 MHz, optionally 160 MHz. It has 17 GPIOs, a single 10-bit ADC whose input range is 0 to 1.0 V, and no Bluetooth. Its boot ROM prints at 74880 baud, which is why the first lines after reset look like garbage at 115200."},
    {"title": "Arduino Mega 2560 Pin Mapping",
     "text": "The Arduino Mega 2560 is based on the ATmega2560 and provides 54 digital I/O pins, of which 15 support PWM, together with 16 analog inputs at 10-bit resolution. It has 256 KB of flash, 8 KB of SRAM and 4 KB of EEPROM, runs at 16 MHz, and offers four hardware UARTs. The default I2C pins are 20 (SDA) and 21 (SCL)."},
    {"title": "Teensy 4.0 Specifications",
     "text": "The Teensy 4.0 uses an NXP i.MX RT1062 with an Arm Cortex-M7 at 600 MHz. It has 1024 KB of RAM, 2 MB of flash, 40 digital pins, 14 analog inputs, 31 PWM outputs, 7 serial ports, 3 SPI and 3 I2C buses, and 3 CAN buses. Logic level is 3.3 V and the pins are not 5 V tolerant."},
    {"title": "MSP430G2553 LaunchPad Guide",
     "text": "The MSP430G2553 is a 16-bit ultra-low-power microcontroller with 16 KB of flash and 512 bytes of RAM, clocked at up to 16 MHz. It runs from 1.8 V to 3.6 V and draws about 0.5 microamps in standby mode. It includes a 10-bit ADC with 8 channels and two 16-bit timers."},
    {"title": "PIC16F877A Reference",
     "text": "The PIC16F877A is an 8-bit microcontroller with 14.3 KB of program memory, 368 bytes of RAM and 256 bytes of EEPROM. It supports clock speeds up to 20 MHz, operates from 2.0 V to 5.5 V, and each I/O pin can source or sink up to 25 mA. It has a 10-bit ADC with 8 channels and 33 I/O pins."},
    {"title": "BBC micro:bit V2 Hardware",
     "text": "The BBC micro:bit V2 is built on the Nordic nRF52833, a Cortex-M4 at 64 MHz with 512 KB of flash and 128 KB of RAM. The edge connector exposes 25 pins, with 19 usable as GPIO. It includes a 5 by 5 LED matrix, an accelerometer, a magnetometer, a microphone and a speaker, and is programmed over a separate USB interface chip."},
    {"title": "Raspberry Pi 4 GPIO Header",
     "text": "The Raspberry Pi 4 Model B has a 40-pin header with 28 GPIO lines running at 3.3 V logic. It has no analog inputs, so an external ADC is required for sensors. The default I2C bus uses GPIO2 (SDA) and GPIO3 (SCL), and the primary UART is on GPIO14 (TXD) and GPIO15 (RXD). The total 3.3 V rail budget is about 500 mA."},
    {"title": "SAMD21 Arduino Zero Notes",
     "text": "The ATSAMD21G18 used on the Arduino Zero is an Arm Cortex-M0+ running at 48 MHz with 256 KB of flash and 32 KB of SRAM. It operates at 3.3 V, each pin is limited to 7 mA, and it offers a 12-bit ADC with 14 channels and a single 10-bit DAC on pin A0. It has a native full-speed USB port."},
    {"title": "CH32V003 Low-Cost RISC-V MCU",
     "text": "The CH32V003 is a RISC-V microcontroller with a QingKe V2A core at 48 MHz, 16 KB of flash and 2 KB of SRAM. It runs from 3.3 V or 5 V, has 18 GPIOs in the TSSOP20 package, a 10-bit ADC with 8 external channels, and is programmed through a single-wire debug interface on pin PD1."},
    {"title": "8051 Architecture Refresher",
     "text": "The classic 8051 is an 8-bit microcontroller with 4 KB of ROM, 128 bytes of RAM, 32 I/O lines arranged as four 8-bit ports, two 16-bit timers and one serial port. A machine cycle takes 12 oscillator periods, so a 12 MHz crystal gives one machine cycle per microsecond. Port 0 requires external pull-up resistors."},
    # --- unrelated general topics: easy distractors -------------------------
    {"title": "Sourdough Starter Maintenance",
     "text": "A sourdough starter should be fed with equal weights of flour and water once a day at room temperature. A healthy starter doubles in volume within four to eight hours of feeding. Kept in the refrigerator it needs feeding only about once a week, and should be refreshed twice before baking."},
    {"title": "The Nile River",
     "text": "The Nile flows northward for about 6,650 kilometres through eleven countries before reaching the Mediterranean Sea. Its two main tributaries, the White Nile and the Blue Nile, meet at Khartoum. The Aswan High Dam, completed in 1970, created Lake Nasser and regulates the annual flood."},
    {"title": "Rules of Badminton Scoring",
     "text": "A badminton match is played as the best of three games to 21 points, with a point scored on every rally. At 20-all a side must lead by two points to win the game, and at 29-all the side scoring the 30th point wins. Players change ends after each game."},
    {"title": "Photosynthesis Summary",
     "text": "Photosynthesis converts carbon dioxide and water into glucose and oxygen using light energy absorbed by chlorophyll. The light-dependent reactions take place in the thylakoid membranes and produce ATP and NADPH, which the Calvin cycle then uses in the stroma to fix carbon."},
    {"title": "History of the Printing Press",
     "text": "Johannes Gutenberg introduced movable-type printing to Europe around 1440 in Mainz. His 42-line Bible, completed in about 1455, was the first major book printed with the technique. By 1500 printing presses across Europe had produced an estimated twenty million volumes."},
    {"title": "Tea Brewing Temperatures",
     "text": "Green tea is best brewed with water at 70 to 80 degrees Celsius for two to three minutes, while black tea needs water close to boiling for three to five minutes. Oolong sits between the two at around 85 to 95 degrees. Water that is too hot makes green tea bitter."},
    {"title": "Marathon Training Basics",
     "text": "Most marathon plans last 16 to 20 weeks and build up to a long run of about 32 kilometres three weeks before the race. Weekly mileage should increase by no more than roughly ten percent. The final two to three weeks are a taper, in which volume drops while intensity is maintained."},
    {"title": "The Periodic Table: Noble Gases",
     "text": "The noble gases occupy group 18 of the periodic table: helium, neon, argon, krypton, xenon and radon. They have full outer electron shells, which makes them largely unreactive. Argon makes up about 0.93 percent of the Earth's atmosphere and is commonly used as a shielding gas in welding."},
    {"title": "Chess Opening Principles",
     "text": "Sound opening play in chess follows three principles: control the centre with pawns, develop knights and bishops before moving the same piece twice, and castle early to protect the king. Bringing the queen out too early usually loses time, because the opponent develops pieces while attacking it."},
    {"title": "Monsoon Climate of South Asia",
     "text": "The southwest monsoon reaches the coast of Kerala around the first of June and covers most of the Indian subcontinent by mid-July. It delivers roughly three quarters of the region's annual rainfall. The retreating monsoon between October and December brings rain to the southeastern coast."},
    {"title": "How Vaccines Train the Immune System",
     "text": "A vaccine presents the immune system with a harmless antigen so that B cells produce antibodies and memory cells are formed. On later exposure to the real pathogen the memory cells respond within days rather than weeks. Booster doses raise the antibody level again when it wanes."},
    {"title": "Basics of Double-Entry Bookkeeping",
     "text": "In double-entry bookkeeping every transaction is recorded as a debit in one account and an equal credit in another, so the ledger always balances. Assets and expenses increase with debits, while liabilities, equity and income increase with credits. A trial balance lists all balances to check for errors."},
    {"title": "Acoustic Guitar String Care",
     "text": "Acoustic guitar strings typically last one to three months with regular playing. Wiping them with a dry cloth after each session slows corrosion. Standard tuning from the lowest string is E, A, D, G, B, E. New strings stretch, so the instrument needs retuning several times in the first day."},
    {"title": "The Great Barrier Reef",
     "text": "The Great Barrier Reef stretches for more than 2,300 kilometres along the coast of Queensland and consists of nearly 3,000 individual reefs. It supports over 1,500 species of fish. Rising sea temperatures have caused several mass coral bleaching events since 1998."},
    {"title": "Introduction to Roman Aqueducts",
     "text": "Roman aqueducts carried water by gravity alone along gradients as gentle as 34 centimetres per kilometre. Eleven aqueducts supplied the city of Rome by the third century. Most of their length ran underground; the famous arched bridges were needed only to cross valleys."},
    {"title": "Composting at Home",
     "text": "A compost heap needs roughly three parts carbon-rich brown material such as dry leaves to one part nitrogen-rich green material such as vegetable scraps. It should be kept as damp as a wrung-out sponge and turned every week or two. Finished compost is dark, crumbly and smells of earth."},
]

_NOISE_GENERATION_PROMPT = """Write {n} short encyclopedia-style passages, each 50 to 80 words long.
Rules:
- Every passage must be about a different topic.
- None may be about ESP32 chips, Espressif, or anything that helps answer this question: "{question}"
- Include concrete numbers so the passages read like reference material.
Format each passage exactly as:
TITLE: <title>
TEXT: <passage>"""

_GENERATED = re.compile(r"TITLE:\s*(.+?)\s*\n+TEXT:\s*(.+?)(?=\n\s*TITLE:|\Z)", re.S)


class NoiseProvider:
    def __init__(self, llm: Optional[LLMClient] = None, source: str = "pool") -> None:
        self.llm = llm
        self.source = source
        self._cache: dict[str, list[dict[str, str]]] = {}
        self._lock = threading.Lock()

    def _generate(self, question: str, n: int) -> list[dict[str, str]]:
        if self.llm is None:
            return []
        with self._lock:
            cached = self._cache.get(question, [])
            if len(cached) >= n:
                return cached[:n]
        try:
            raw = self.llm.chat(
                [{"role": "user", "content": _NOISE_GENERATION_PROMPT.format(n=n, question=question)}],
                max_tokens=min(140 * n + 60, 1500),
                temperature=0.8,
            )
        except LLMError:
            return cached
        generated = [
            {"title": title.strip()[:120], "text": " ".join(text.split())}
            for title, text in _GENERATED.findall(raw)
            if len(text.split()) >= 20 and "esp32" not in text.lower()
        ]
        with self._lock:
            self._cache[question] = (cached + generated)[: max(n, len(cached))]
            return self._cache[question][:n]

    def sample(
        self,
        n: int,
        rng: random.Random,
        question: str,
        origin: NoiseOrigin,
        exclude_titles: Optional[set[str]] = None,
    ) -> list[ChunkTrace]:
        """Return `n` distinct noise documents, none of which is in `exclude_titles`."""
        if n <= 0:
            return []
        exclude = set(exclude_titles or ())
        picked: list[dict[str, str]] = []
        generated_count = 0
        if self.source == "llm":
            for doc in self._generate(question, n):
                if doc["title"] not in exclude:
                    picked.append(doc)
                    exclude.add(doc["title"])
            generated_count = len(picked)
        remaining = [doc for doc in NOISE_POOL if doc["title"] not in exclude]
        rng.shuffle(remaining)
        picked.extend(remaining[: max(n - len(picked), 0)])

        chunks: list[ChunkTrace] = []
        for index, doc in enumerate(picked[:n]):
            generated = index < generated_count
            chunks.append(
                ChunkTrace(
                    source_id="",
                    chunk_id=f"noise-{origin}-{hashlib.sha1(doc['title'].encode('utf-8')).hexdigest()[:10]}",
                    title=doc["title"],
                    chip="noise",
                    section="LLM-generated distractor" if generated else "Distractor pool",
                    source_type="noise",
                    text=doc["text"],
                    is_noise=True,
                    noise_origin=origin,
                )
            )
        return chunks
