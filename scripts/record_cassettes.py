"""Record ClinicalTrials.gov responses into tests/cassettes/ctgov for offline replay.

Usage:  python scripts/record_cassettes.py [--delay 0.5]

Each search is recorded with and without its country filter (page size 10), then
the eligibility record for every NCT id returned. CT.gov is a public service, so
calls are made sequentially with a small delay between them.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from trialmatch.ports import TrialNotFound, TrialsUnavailable  # noqa: E402
from trialmatch.trials import CtGovClient  # noqa: E402

SEARCHES: list[tuple[str, list[str]]] = [
    ("type 2 diabetes", ["Netherlands", "United States"]),
    ("breast cancer", ["United States", "Netherlands"]),
    ("COPD", ["Netherlands"]),
    ("heart failure", ["United States"]),
    ("asthma", ["Netherlands"]),
    ("rheumatoid arthritis", ["Germany"]),
]
PAGE_SIZE = 10


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--delay", type=float, default=0.5, help="seconds between requests")
    ap.add_argument("--cassette-dir", type=Path, default=ROOT / "tests" / "cassettes" / "ctgov")
    args = ap.parse_args()

    client = CtGovClient(mode="record", cassette_dir=args.cassette_dir)
    nct_ids: set[str] = set()
    requests = 0
    with client:
        for condition, countries in SEARCHES:
            for country in [None, *countries]:
                hits = client.search(condition, country=country, page_size=PAGE_SIZE)
                requests += 1
                nct_ids.update(h.nct_id for h in hits)
                print(f"search {condition!r} country={country!r}: {len(hits)} hits")
                time.sleep(args.delay)
        for nid in sorted(nct_ids):
            try:
                client.eligibility(nid)
                print(f"eligibility {nid}: ok")
            except (TrialNotFound, TrialsUnavailable) as exc:
                print(f"eligibility {nid}: {exc}")
            requests += 1
            time.sleep(args.delay)
    print(f"done: {requests} requests, {len(nct_ids)} unique trials")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
