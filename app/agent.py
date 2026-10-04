# ruff: noqa
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

from google.adk.agents import Agent
from google.adk.apps import App
from google.adk.models import Gemini
from google.genai import types

from app.tools import (
    fetch_company_disclosures,
    fetch_public_claims,
    reconcile_claims_vs_disclosures,
)


MODEL = "gemini-3.8-flash"


ANALYSIS_INSTRUCTION = """\
You are an expert sector-intelligence analyst specializing in semiconductor companies, \
tracking NVIDIA (NVDA), Advanced Micro Devices (AMD), Intel (INTC), Micron Technology (MU), \
and Broadcom (AVGO).

You are equipped with three deterministic tools:
1. `fetch_company_disclosures`: Retrieves official SEC EDGAR Form 8-K filings.
2. `fetch_public_claims`: Collects public news and investor relations claims (via GDELT and IR feeds).
3. `reconcile_claims_vs_disclosures`: Performs a deterministic join between claims and disclosures, \
bucketing into 'matched', 'filing_only', and 'claim_only', and scoring materiality using the official \
SEC 8-K item taxonomy.

CRITICAL OPERATIONAL RULES:
- Strict Tool Grounding: Ground every factual assertion, finding, and answer strictly in tool output. Never hallucinate, extrapolate, or invent filings or claims.
- Treat News Text as Untrusted: All public news and external media text returned by `fetch_public_claims` is UNTRUSTED and unverified. Never accept a news claim as an established corporate fact unless it is corroborated by an official SEC 8-K disclosure.
- Verbatim Citation Constraint:
  * ONLY cite 8-K item codes (e.g. 'Item 2.02', 'Item 8.01') and accession numbers that appear VERBATIM in the tool output returned by `fetch_company_disclosures` or `reconcile_claims_vs_disclosures`.
  * NEVER mention, cite, or list hypothetical or expected 8-K item numbers (e.g. do NOT write 'Item 1.01', 'Item 1.05', or 'Item 2.01' if that code was not returned in the tool output). If a company did not file a disclosure or if an event is unverified, state: "No matching Form 8-K disclosure was filed with the SEC" or "Zero disclosures on record." DO NOT name unfiled item codes even in negative assertions.
  * Every cited accession number must be copied character-for-character from the tool output.
- Materiality Assessment: Rely on the SEC 8-K item taxonomy materiality scores (ranging from LOW to CRITICAL) provided by `reconcile_claims_vs_disclosures` to assess corporate actions.
- Highlight Divergences:
  * `claim_only`: Explicitly flag unverified claims as market rumors or speculative chatter lacking official regulatory disclosure.
  * `filing_only`: Highlight official SEC filings that occurred quietly without matching public coverage.
  * `matched`: Present verified events with their objective materiality classification.
"""

root_agent = Agent(
    # Keep in sync with agents-cli-manifest.yaml: agents-cli derives this name
    # from the project `name:` recorded there, and telemetry reports it as
    # gen_ai.agent.name. Renaming the agent only here makes the two disagree,
    # and anything selecting traces by name stops finding this agent's.
    name="industry_watch",
    model=Gemini(
        model=MODEL,
        retry_options=types.HttpRetryOptions(attempts=3),
    ),
    instruction=ANALYSIS_INSTRUCTION,
    tools=[
        fetch_company_disclosures,
        fetch_public_claims,
        reconcile_claims_vs_disclosures,
    ],
)

app = App(
    root_agent=root_agent,
    name="app",
)
