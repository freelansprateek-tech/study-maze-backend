"""Fast unit tests for the parts of the pipeline that need no model or database.

    pytest -q
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.chips import CHIP_IDS  # noqa: E402
from app.core import assembler  # noqa: E402
from app.core.classifier import kmeans_split  # noqa: E402
from app.core.pipeline import noise_slots  # noqa: E402
from app.core.processors import (  # noqa: E402
    extract_hardware_entities, extractive_summary, numbers_are_grounded, parse_mso_output,
)
from app.core.retriever import ChipFilteredRetriever  # noqa: E402
from app.ingest.chunker import chunk_section  # noqa: E402
from app.ingest.parsers import parse_json, parse_markdown, parse_rst, parse_txt  # noqa: E402
from app.ingest.tagger import chip_from_path, tag_chunk  # noqa: E402
from app.schemas import PromptItem  # noqa: E402
from eval.metrics import exact_match, f1_score  # noqa: E402


def test_tagger_distinguishes_variants_from_classic():
    assert tag_chunk("The ESP32-S3 has 45 GPIOs.") == "esp32-s3"
    assert tag_chunk("The ESP32 has 34 GPIOs. The ESP32 is dual-core.") == "esp32"
    assert tag_chunk("ESP32, ESP32-S3 and ESP32-C3 all support Wi-Fi.") == "common"
    assert tag_chunk("Call esp_wifi_init() first.") == "common"
    assert tag_chunk("anything", explicit_chip="ESP32C6") == "esp32-c6"
    assert chip_from_path("data/raw/esp32-c3/gpio.rst") == "esp32-c3"
    assert chip_from_path("docs/esp32s2_datasheet_en.pdf") == "esp32-s2"
    assert chip_from_path("data/raw/common/wifi.md") == "common"


def test_chip_filter_is_selected_chip_plus_common():
    assert ChipFilteredRetriever.chip_filter("esp32-c6") == {"chip": {"$in": ["esp32-c6", "common"]}}


def test_parsers(tmp_path: Path):
    md = tmp_path / "gpio.md"
    md.write_text("# GPIO Guide\n\nIntro text here.\n\n## Pull-ups\n\nUse `gpio_config()` with [docs](http://x).\n")
    doc = parse_markdown(md)[0]
    assert doc.title == "GPIO Guide" and [s.heading for s in doc.sections] == ["GPIO Guide", "Pull-ups"]
    assert "http" not in doc.sections[1].text

    rst = tmp_path / "adc.rst"
    rst.write_text("ADC Driver\n==========\n\n{IDF_TARGET_NAME} has :cpp:func:`adc_oneshot_read`.\n\nLimits\n------\n\n.. note::\n\n    ADC2 is shared.\n")
    doc = parse_rst(rst, target_chip="esp32-s3")[0]
    assert doc.title == "ADC Driver" and "ESP32-S3 has adc_oneshot_read" in doc.sections[0].text
    assert doc.sections[1].heading == "Limits" and "ADC2 is shared" in doc.sections[1].text

    js = tmp_path / "issues.json"
    js.write_text(json.dumps([{"number": 7, "title": "ADC broken", "body": "It fails on my board.", "html_url": "http://h",
                               "comments": [{"user": {"login": "bob"}, "body": "Use ADC1."}]}]))
    doc = parse_json(js)[0]
    assert doc.source_type == "github_issue" and doc.url == "http://h" and "bob wrote: Use ADC1." in doc.sections[1].text

    txt = tmp_path / "thread.txt"
    txt.write_text("Title: Board resets\n\nIt resets when WiFi starts.\n")
    assert parse_txt(txt)[0].title == "Board resets"


def test_chunker_respects_limit_and_overlaps():
    text = "\n\n".join(" ".join(f"w{p}_{i}" for i in range(60)) for p in range(5))
    chunks = chunk_section(text, chunk_words=150, overlap_words=20)
    assert len(chunks) >= 2 and all(len(c.split()) <= 170 for c in chunks)
    assert chunks[0].split()[-1] in chunks[1].split()[:20]


def test_kmeans_split_higher_centroid_is_direct():
    labels, direct_c, indirect_c = kmeans_split([0.9, 0.85, 0.1, 0.12, 0.08])
    assert labels == ["direct", "direct", "indirect", "indirect", "indirect"] and direct_c > indirect_c
    assert kmeans_split([0.5])[0] == ["direct"]
    assert kmeans_split([0.3, 0.3, 0.3])[0] == ["direct"] * 3


def test_noise_slots():
    assert noise_slots(8, 0.0) == (8, 0)
    assert noise_slots(8, 0.4) == (5, 3)
    assert noise_slots(8, 0.8) == (2, 6)
    assert noise_slots(1, 0.8) == (1, 0)


def test_entity_extraction_and_grounding_guard():
    text = "ADC1 is on GPIO32 to GPIO39. Deep-sleep draws 10 µA. Call adc_oneshot_read() and set CONFIG_FREERTOS_HZ."
    found = {e.text: e.type for e in extract_hardware_entities(text)}
    assert found.get("10 µA") == "electrical" and found.get("adc_oneshot_read()") == "api"
    assert found.get("CONFIG_FREERTOS_HZ") == "config" and any(t == "pin" for t in found.values())
    assert numbers_are_grounded("It draws 10 µA.", text, "")
    assert not numbers_are_grounded("It draws 25 µA.", text, "")
    assert "10 µA" in extractive_summary("What is the deep-sleep current?", text, max_sentences=1)


def test_parse_mso_output():
    raw = "ENTITIES: GPIO25; GPIO26 (pin)\nRELATIONS: DAC1 | is on | GPIO25; bad\nSUMMARY: DAC outputs are GPIO25 and GPIO26."
    entities, relations, summary = parse_mso_output(raw)
    assert entities == ["GPIO25", "GPIO26"] and relations[0].object == "GPIO25" and summary.startswith("DAC outputs")
    assert parse_mso_output("just prose")[2] == "just prose"


def test_prompt_layout_is_head_middle_tail_question():
    direct = [PromptItem(source_id=f"S{i}", kind="direct", title=f"D{i}", text=f"direct {i}") for i in (1, 2, 3)]
    middle = [PromptItem(source_id="S9", kind="noise", title="N", text="noise")]
    messages, layout = assembler.assemble_prompt("esp32-c3", "Which pin?", direct, middle, {})
    assert [s.position for s in layout] == ["system", "head", "middle", "tail", "question"]
    assert [i.source_id for i in layout[1].items] == ["S1", "S2"] and [i.source_id for i in layout[3].items] == ["S3"]
    user = messages[1]["content"]
    assert user.index("direct 1") < user.index("noise") < user.index("direct 3") < user.index("Which pin?")
    assert user.rstrip().endswith(assembler.COT_TRIGGER)
    assert "ESP32-C3" in messages[0]["content"] and "does not cover this detail" in messages[0]["content"]

    _, flat = assembler.assemble_prompt("esp32", "q?", direct, middle, {}, u_shape=False, flat_order=["S9", "S1", "S2", "S3"])
    assert [s.position for s in flat] == ["system", "flat", "question"]
    assert [i.source_id for i in flat[1].items] == ["S9", "S1", "S2", "S3"]


def test_parse_answer_statuses():
    ids = {"S1", "S2"}
    answer, _explanation, reasoning, cited, status = assembler.parse_answer("GPIO9 is the boot pin [S1].\nFinal answer: GPIO9 [S1, S7]", "esp32-c3", ids)
    assert answer == "GPIO9 [S1, S7]" and cited == ["S1"] and status == "grounded" and "boot pin" in reasoning
    assert assembler.parse_answer("Final answer: GPIO9", "esp32-c3", ids)[4] == "uncited"
    answer, explanation, _, cited, status = assembler.parse_answer(
        "Step [S1].\nFinal answer: GPIO9 is the boot pin [S1].\nExplanation: Hold it low at reset [S7]. It has 99 modes.", "esp32-c3", ids)
    assert answer == "GPIO9 is the boot pin [S1]." and explanation.startswith("Hold it low") and status == "grounded"
    assert assembler.keep_supported_sentences(explanation, ["GPIO9 hold low at reset"]) == "Hold it low at reset [S7]."
    answer, _, _, cited, status = assembler.parse_answer("The documentation for the ESP32-C3 does not cover this detail.", "esp32-c3", ids)
    assert status == "not_covered" and cited == [] and answer == assembler.not_covered_statement("esp32-c3")


def test_metrics_are_boundary_aware():
    assert exact_match("It has 6 channels [S1].", ["6"]) == 1.0
    assert exact_match("It has 16 channels on GPIO6.", ["6"]) == 0.0
    assert exact_match("offset 0x1000", ["0x0"]) == 0.0
    assert exact_match("about 10 µA in deep sleep", ["10 uA"]) == 1.0
    assert exact_match("two 13-bit SAR ADCs", ["13 bits"]) == 1.0
    assert exact_match("GPIO25 only", ["GPIO25 and GPIO26"], all_of=["GPIO25", "GPIO26"]) == 0.0
    assert exact_match("GPIO26 and GPIO25", ["x"], all_of=["GPIO25", "GPIO26"]) == 1.0
    assert exact_match("The outputs are on GPIO 25 and GPIO 26.", ["x"], all_of=["GPIO25", "GPIO26"]) == 1.0
    assert exact_match("It has up to seven ADC channels [S1].", ["7"]) == 1.0
    assert f1_score("GPIO9 [S1]", ["GPIO9"]) == 1.0 and f1_score("pin GPIO9", ["GPIO9"]) < 1.0


def test_seed_corpus_and_benchmark_are_consistent():
    seed = json.loads((ROOT / "data" / "seed" / "seed_corpus.json").read_text())["records"]
    ids = {r["id"] for r in seed}
    assert len(seed) >= 35 and len(ids) == len(seed)
    assert {r["chip"] for r in seed} == set(CHIP_IDS) | {"common"}
    bench = json.loads((ROOT / "eval" / "esp32_qa_benchmark.json").read_text())["items"]
    assert len(bench) >= 25 and {b["chip"] for b in bench} == set(CHIP_IDS)
    by_id = {r["id"]: r for r in seed}
    for item in bench:
        source = by_id[item["source_chunk_id"]]
        assert source["chip"] == item["chip"]
        # every gold answer must literally be supported by its source chunk
        assert exact_match(source["text"], item["answers"], item.get("all_of")) == 1.0, item["id"]


class _FakeLLM:
    provider, model, parallel = "fake", "fake", False

    def __init__(self, reply: str) -> None:
        self.reply = reply

    def chat(self, messages, max_tokens, temperature=None):
        return self.reply


def _chunk(text: str, title: str = "ESP32-C6 Series Datasheet"):
    from app.schemas import ChunkTrace

    return ChunkTrace(source_id="S1", chunk_id="c1", title=title, chip="esp32-c6", text=text)


def test_direct_processor_guards():
    from app.core.processors import DirectAssistProcessor, render_direct_item

    text = "The ESP32-C6 Wi-Fi radio operates in the 2.4 GHz band only; the 5 GHz band is not supported. It supports 802.11ax."
    question = "Does the chip support 5 GHz Wi-Fi?"

    # An LLM summary is accepted, and is sent together with verbatim evidence.
    ok = DirectAssistProcessor(_FakeLLM("ENTITIES: 5 GHz\nRELATIONS:\nSUMMARY: The 5 GHz band is not supported."))
    summary = ok.process(question, "ESP32-C6", [_chunk(text)])[0]
    assert summary.method == "llm" and summary.relevant and "5 GHz band is not supported" in summary.evidence
    rendered = render_direct_item(summary, text)
    assert rendered.startswith("Summary: ") and f"Source text: {text}" in rendered

    # A summary that drops the source's negation is rejected, and so is the relation that does.
    flipped = DirectAssistProcessor(_FakeLLM(
        "ENTITIES: Wi-Fi; 5 GHz\nRELATIONS: 5 GHz band | is | supported; Wi-Fi radio | operates in | 2.4 GHz band only\n"
        "SUMMARY: The ESP32-C6 Wi-Fi radio supports the 5 GHz band."))
    result = flipped.process(question, "ESP32-C6", [_chunk(text)])[0]
    assert result.method == "extractive" and "not supported" in result.summary
    assert [r.subject for r in result.relations] == ["Wi-Fi radio"]

    # A summary that answers the question with words the chunk never uses is rejected.
    invented = DirectAssistProcessor(_FakeLLM("SUMMARY: GPIO0 selects the flash regulator and boots from the bootloader partition."))
    assert invented.process(question, "ESP32-C6", [_chunk(text)])[0].method == "extractive"

    # An invented number is rejected in favour of sentences copied from the chunk.
    bad = DirectAssistProcessor(_FakeLLM("SUMMARY: It supports the 6 GHz band at 160 MHz."))
    assert bad.process(question, "ESP32-C6", [_chunk(text)])[0].method == "extractive"

    # "IRRELEVANT" is overruled when the chunk plainly shares the question's vocabulary...
    wrong = DirectAssistProcessor(_FakeLLM("ENTITIES:\nRELATIONS:\nSUMMARY: IRRELEVANT"))
    kept = wrong.process(question, "ESP32-C6", [_chunk(text)])[0]
    assert kept.relevant and kept.method == "extractive" and "not supported" in kept.summary
    # ...and honoured when it does not.
    noise = "A sourdough starter should be fed with equal weights of flour and water once a day at room temperature."
    assert not wrong.process(question, "ESP32-C6", [_chunk(noise, "Sourdough Starter Maintenance")])[0].relevant


def test_unsupported_values_guard():
    sources = ["ESP32-C6 Series Datasheet The radio operates in the 2.4 GHz band only; the 5 GHz band is not supported. "
               "GPIO9 selects the boot mode. Deep-sleep draws 7 µA. Offset 0x1000. Two 12-bit ADCs.", "ESP32-C6"]
    ok = assembler.unsupported_values
    assert ok("The ESP32-C6 does not support 5 GHz Wi-Fi [S5].", sources) == []
    assert ok("GPIO9 [S2]; it draws 7 uA; offset 0x1000; 12 bits", sources) == []
    assert ok("It does not support 5.5 GHz Wi-Fi. [S5]", sources) == ["5.5 GHz"]
    assert ok("10 MHz [S2]", sources) == ["10 MHz"]
    assert ok("Use GPIO19 for this.", sources) == ["GPIO19"]
    assert ok("1. It is GPIO9.\n2. Hold it low.", sources) == []
