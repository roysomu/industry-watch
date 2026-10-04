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

"""Unit tests for deterministic tools in app.tools."""

from app.tools import (
    TICKER_TO_CIK,
    fetch_company_disclosures,
    fetch_public_claims,
    reconcile_claims_vs_disclosures,
)


def test_tracked_tickers_have_ciks():
    """Ensure all five requested semiconductor tickers have mapped CIKs."""
    required = {"NVDA", "AMD", "INTC", "MU", "AVGO"}
    assert required.issubset(set(TICKER_TO_CIK.keys()))
    for t in required:
        assert len(TICKER_TO_CIK[t]) == 10


def test_fetch_company_disclosures_unsupported_ticker():
    """Test error handling for non-tracked ticker."""
    res = fetch_company_disclosures("UNKNOWN_TICKER")
    assert res["status"] == "error"
    assert res["error_code"] == "UNSUPPORTED_TICKER"
    assert res["filings"] == []


def test_fetch_public_claims_untrusted_formatting():
    """Verify that public claims are explicitly formatted with untrusted boundaries."""
    claims_res = fetch_public_claims("NVDA", query="Blackwell", limit=2)
    assert claims_res["status"] == "success"
    assert "claims" in claims_res
    for claim in claims_res["claims"]:
        assert "claim_text_untrusted" in claim
        assert "<UNTRUSTED_PUBLIC_CLAIM" in claim["claim_text_untrusted"]
        assert claim["trust_level"] in (
            "UNTRUSTED_EXTERNAL_SOURCE",
            "UNVERIFIED_IR_STATEMENT",
        )


def test_reconcile_claims_vs_disclosures_buckets():
    """Test deterministic join, bucketing, and materiality scoring."""
    sample_filings = [
        {
            "filing_id": "0001-filing-match",
            "ticker": "NVDA",
            "filing_date": "2026-08-20",
            "items": ["2.02", "9.01"],  # Earnings: Item 2.02 (Materiality 0.90 / HIGH)
            "document_url": "https://sec.gov/filing1",
        },
        {
            "filing_id": "0002-filing-only",
            "ticker": "NVDA",
            "filing_date": "2026-07-10",
            "items": ["1.05"],  # Cybersecurity: Item 1.05 (Materiality 1.00 / CRITICAL)
            "document_url": "https://sec.gov/filing2",
        },
    ]

    sample_claims = [
        {
            "claim_id": "claim-match-1",
            "ticker": "NVDA",
            "publish_date": "2026-08-21",  # Within 3 days of 2026-08-20
            "source_name": "Reuters",
            "title": "Nvidia reports record earnings",
            "claim_text_untrusted": "<UNTRUSTED>Nvidia reports record earnings</UNTRUSTED>",
        },
        {
            "claim_id": "claim-only-1",
            "ticker": "NVDA",
            "publish_date": "2026-05-01",  # Distant date, no matching filing
            "source_name": "TechBlog",
            "title": "Rumor: Nvidia to acquire startup",
            "claim_text_untrusted": "<UNTRUSTED>Rumor: Nvidia to acquire startup</UNTRUSTED>",
        },
    ]

    result = reconcile_claims_vs_disclosures(
        disclosures=sample_filings,
        claims=sample_claims,
        window_days=3,
    )

    assert result["status"] == "success"
    summary = result["summary"]
    results = result["reconciliation_results"]

    # Verify bucket counts
    assert summary["matched_events_count"] == 1
    assert summary["filing_only_count"] == 1
    assert summary["claim_only_unverified_count"] == 1

    # Verify matched event details
    matched_event = results["matched"][0]
    assert matched_event["filing_id"] == "0001-filing-match"
    assert matched_event["materiality_score"] == 0.90
    assert matched_event["materiality_level"] == "HIGH"
    assert len(matched_event["associated_public_claims"]) == 1

    # Verify filing-only details
    filing_only_event = results["filing_only"][0]
    assert filing_only_event["filing_id"] == "0002-filing-only"
    assert filing_only_event["materiality_score"] == 1.00
    assert filing_only_event["materiality_level"] == "CRITICAL"

    # Verify claim-only details
    claim_only_event = results["claim_only"][0]
    assert claim_only_event["claim_id"] == "claim-only-1"
    assert claim_only_event["trust_level"] == "UNTRUSTED_SPECULATION"
