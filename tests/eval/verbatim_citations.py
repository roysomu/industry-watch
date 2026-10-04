# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Deterministic evaluation metric: Verbatim citation of accession numbers and 8-K items."""

import json
import re
from typing import Any


def _extract_all_tool_outputs(instance: dict[str, Any]) -> str:
    """Collect all tool response payloads across all turns into a single search string."""
    tool_text_chunks: list[str] = []
    agent_data = instance.get("agent_data") or {}
    turns = agent_data.get("turns") or []

    for turn in turns:
        events = turn.get("events") or []
        for event in events:
            # Check content parts
            content = event.get("content") or {}
            parts = content.get("parts") or []
            for part in parts:
                if "function_response" in part:
                    resp = part["function_response"].get("response", {})
                    tool_text_chunks.append(json.dumps(resp))
            # Also check direct tool output fields if present
            if event.get("author") == "tool" or event.get("role") == "tool":
                tool_text_chunks.append(json.dumps(event))

    return " ".join(tool_text_chunks)


def _extract_all_agent_text(instance: dict[str, Any]) -> str:
    """Collect all agent text responses across turns."""
    agent_texts: list[str] = []

    # Final response
    resp = instance.get("response") or {}
    for part in resp.get("parts") or []:
        if "text" in part:
            agent_texts.append(part["text"])

    # All model turns
    turns = (instance.get("agent_data") or {}).get("turns") or []
    for turn in turns:
        for event in turn.get("events") or []:
            if event.get("author") != "user" and event.get("role") in ("model", "assistant"):
                for part in (event.get("content") or {}).get("parts") or []:
                    if "text" in part:
                        agent_texts.append(part["text"])

    return "\n".join(agent_texts)


def evaluate(instance: dict[str, Any]) -> dict[str, Any]:
    """Deterministic evaluation of citations against tool outputs.

    Rule: Every accession number (e.g. 0001045810-26-000078) and 8-K item code
    (e.g. Item 2.02, Item 8.01) cited in the agent's text must appear verbatim
    in the tool output.
    """
    agent_text = _extract_all_agent_text(instance)
    tool_output_str = _extract_all_tool_outputs(instance)

    # 1. Accession numbers pattern: 10 digits - 2 digits - 6 digits
    accession_pattern = re.compile(r"\b\d{10}-\d{2}-\d{6}\b")
    cited_accessions = set(accession_pattern.findall(agent_text))

    # 2. 8-K item codes: "Item X.XX" or "Item X.XX,"
    item_pattern = re.compile(r"\bItem\s+([1-9]\.[0-9]{2})\b", re.IGNORECASE)
    cited_items = set(item_pattern.findall(agent_text))

    unmatched_accessions: list[str] = []
    for acc in cited_accessions:
        if acc not in tool_output_str:
            unmatched_accessions.append(acc)

    unmatched_items: list[str] = []
    for itm in cited_items:
        # Check either '2.02' or 'Item 2.02' in tool output
        if itm not in tool_output_str and f"Item {itm}" not in tool_output_str:
            unmatched_items.append(f"Item {itm}")

    total_citations = len(cited_accessions) + len(cited_items)
    violations = len(unmatched_accessions) + len(unmatched_items)

    if violations > 0:
        return {
            "score": 0.0,
            "explanation": (
                f"FAILED verbatim check. Unmatched citations not in tool output: "
                f"accessions={unmatched_accessions}, items={unmatched_items}."
            ),
        }

    return {
        "score": 1.0,
        "explanation": (
            f"PASSED verbatim check. Verified {len(cited_accessions)} accession number(s) "
            f"and {len(cited_items)} 8-K item code(s) against tool outputs."
        ),
    }
