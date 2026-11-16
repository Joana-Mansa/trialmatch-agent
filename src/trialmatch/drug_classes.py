"""A small drug-class table, so code can check medication evidence.

The model often cites a medication to show a criterion like "prior CDK4/6 inhibitor"
is met. Code cannot read criteria in general, but it can check this narrow claim: if
the criterion names a drug class, and every cited medication is a known drug outside
that class, the citation does not support the claim.

The table is deliberately small and conservative. A medication the table does not
know is never flagged, and a criterion that names no known class or drug is skipped.
A deployment would replace this with an RxNorm/ATC lookup behind the same function.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class DrugClass:
    name: str
    aliases: tuple[str, ...]  # phrases that name the class in criteria text
    drugs: tuple[str, ...]  # generic names, lower case


_ENDOCRINE = ("tamoxifen", "toremifene", "letrozole", "anastrozole", "exemestane",
              "fulvestrant", "elacestrant", "goserelin", "leuprolide")
_ANTI_HER2 = ("trastuzumab", "pertuzumab", "lapatinib", "tucatinib", "neratinib",
              "margetuximab")
_CHEMO = ("doxorubicin", "epirubicin", "cyclophosphamide", "paclitaxel", "docetaxel",
          "capecitabine", "carboplatin", "cisplatin", "oxaliplatin", "gemcitabine",
          "eribulin", "fluorouracil", "vinorelbine", "methotrexate", "pemetrexed")

CLASSES: tuple[DrugClass, ...] = (
    DrugClass("CDK4/6 inhibitor", ("cdk4/6", "cdk 4/6", "cdk4 / 6", "cdk4/cdk6",
                                   "cyclin-dependent kinase 4/6", "cyclin dependent kinase 4/6",
                                   "cyclin-dependent kinase 4 and 6"),
              ("palbociclib", "ribociclib", "abemaciclib")),
    DrugClass("aromatase inhibitor", ("aromatase inhibitor",),
              ("letrozole", "anastrozole", "exemestane")),
    DrugClass("SERM", ("serm", "selective estrogen receptor modulator"),
              ("tamoxifen", "toremifene", "raloxifene")),
    DrugClass("SERD", ("serd", "selective estrogen receptor degrader"),
              ("fulvestrant", "elacestrant")),
    DrugClass("endocrine therapy", ("endocrine therapy", "hormonal therapy", "hormone therapy",
                                    "endocrine treatment"), _ENDOCRINE),
    DrugClass("anti-HER2 therapy", ("anti-her2", "her2-targeted", "her2 targeted",
                                    "her2-directed"), _ANTI_HER2),
    DrugClass("anthracycline", ("anthracycline",), ("doxorubicin", "epirubicin")),
    DrugClass("taxane", ("taxane",), ("paclitaxel", "docetaxel")),
    DrugClass("platinum", ("platinum",), ("carboplatin", "cisplatin", "oxaliplatin")),
    DrugClass("chemotherapy", ("chemotherapy", "cytotoxic"), _CHEMO),
    DrugClass("PARP inhibitor", ("parp inhibitor", "parpi"),
              ("olaparib", "talazoparib", "niraparib", "rucaparib")),
    DrugClass("immune checkpoint inhibitor",
              ("checkpoint inhibitor", "anti-pd-1", "anti-pd-l1", "pd-1", "pd-l1", "ctla-4"),
              ("pembrolizumab", "nivolumab", "atezolizumab", "durvalumab", "ipilimumab")),
    DrugClass("insulin", ("insulin",), ("insulin",)),
    DrugClass("SGLT2 inhibitor", ("sglt2", "sglt-2"),
              ("empagliflozin", "dapagliflozin", "canagliflozin", "ertugliflozin")),
    DrugClass("GLP-1 receptor agonist", ("glp-1", "glp1"),
              ("semaglutide", "liraglutide", "dulaglutide", "exenatide", "tirzepatide")),
    DrugClass("TNF inhibitor", ("tnf inhibitor", "anti-tnf", "tnf-alpha", "tnf alpha"),
              ("adalimumab", "etanercept", "infliximab", "certolizumab", "golimumab")),
    DrugClass("JAK inhibitor", ("jak inhibitor",),
              ("tofacitinib", "baricitinib", "upadacitinib", "filgotinib")),
    DrugClass("systemic corticosteroid",
              ("systemic corticosteroid", "systemic steroid", "oral corticosteroid",
               "oral steroid"),
              ("prednisone", "prednisolone", "methylprednisolone", "dexamethasone",
               "hydrocortisone")),
    DrugClass("anticoagulant", ("anticoagulant", "anticoagulation"),
              ("warfarin", "apixaban", "rivaroxaban", "dabigatran", "edoxaban", "heparin",
               "enoxaparin")),
)

KNOWN_DRUGS: frozenset[str] = frozenset(d for c in CLASSES for d in c.drugs) | frozenset(
    ("metformin", "lisinopril", "furosemide", "tiotropium", "albuterol", "amlodipine",
     "carvedilol", "spironolactone", "budesonide", "folic acid")
)


def _has_phrase(text: str, phrase: str) -> bool:
    # "s?" so "aromatase inhibitors" matches the alias "aromatase inhibitor".
    return re.search(rf"(?<![a-z0-9]){re.escape(phrase)}s?(?![a-z0-9])", text) is not None


def drugs_in(text: str) -> set[str]:
    """Known generic drug names found in ``text`` (case-insensitive, whole words)."""
    low = text.lower()
    return {d for d in KNOWN_DRUGS if _has_phrase(low, d)}


def classes_named(text: str) -> list[DrugClass]:
    """Drug classes a criterion names, by alias."""
    low = text.lower()
    return [c for c in CLASSES if any(_has_phrase(low, a) for a in c.aliases)]


def unsupported_medication_claim(criterion: str, cited_medications: list[str]) -> str | None:
    """Explain why the cited medications cannot support a "met" claim, or return None.

    ``cited_medications`` are the display texts of the medications cited as evidence.
    Returns None (no finding) whenever the table cannot decide.
    """
    named = classes_named(criterion)
    named_drugs = drugs_in(criterion)
    if not cited_medications or not (named or named_drugs):
        return None
    allowed = named_drugs.union(*(set(c.drugs) for c in named))
    cited: set[str] = set()
    for display in cited_medications:
        found = drugs_in(display)
        if not found:
            return None  # a medication the table does not know: cannot judge
        cited |= found
    if cited & allowed:
        return None
    wanted = ", ".join([c.name for c in named] + sorted(named_drugs))
    return (f"cites {', '.join(sorted(cited))}, which is not in the class the criterion "
            f"names ({wanted})")
