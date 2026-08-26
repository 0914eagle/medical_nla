from __future__ import annotations

import unittest

from scripts.make_direct_patient_pdd_splits import build_splits, select_eligible_rows
from scripts.make_direct_canonical_manifest import contains_label


def row(
    index: int,
    *,
    patient: str,
    pdd: str,
    category: str,
    digest: str | None = None,
    resolved: bool = True,
    parsed: bool = True,
):
    return {
        "id": f"r{index:03d}",
        "patient_group": patient,
        "canonical_pdd": pdd,
        "disease_category": category,
        "input_digest": digest or f"d{index:03d}",
        "canonical_pdd_resolved": resolved,
        "patient_id_parsed": parsed,
    }


class DirectPatientPddSplitTest(unittest.TestCase):
    def test_gold_label_phrase_matching_uses_word_boundaries(self):
        self.assertTrue(contains_label("Assessment includes low-risk PE.", "Low-risk PE"))
        self.assertTrue(contains_label("Possible PE was documented.", "PE"))
        self.assertFalse(contains_label("The patient appears stable.", "PE"))
        self.assertFalse(contains_label("History of HFrEF.", "HFpEF"))

    def test_eligibility_excludes_conflicts_unparsed_and_duplicate_copy(self):
        rows = [
            row(1, patient="p1", pdd="A", category="C", digest="same"),
            row(2, patient="p2", pdd="A", category="C", digest="same"),
            row(3, patient="p3", pdd="A", category="C", resolved=False),
            row(4, patient="p4", pdd="A", category="C", parsed=False),
        ]
        eligible, excluded = select_eligible_rows(rows)
        self.assertEqual([item["id"] for item in eligible], ["r001"])
        self.assertEqual(excluded["r002"], "duplicate_copy")
        self.assertEqual(excluded["r003"], "label_conflict")
        self.assertEqual(excluded["r004"], "unparsed_patient")

    def test_split_is_reproducible_patient_disjoint_and_label_disjoint(self):
        rows = []
        index = 0
        for pdd_index, pdd in enumerate("ABCDEF"):
            category = "cardio" if pdd_index < 3 else "neuro"
            for patient_index in range(6):
                index += 1
                rows.append(
                    row(
                        index,
                        patient=f"{pdd}-p{patient_index}",
                        pdd=pdd,
                        category=category,
                    )
                )

        first, _, components_first = build_splits(
            rows,
            seed=17,
            heldout_fraction=0.2,
            train_fraction=0.7,
            val_fraction=0.15,
            min_heldout_label_rows=3,
            min_remaining_category_rows=3,
        )
        second, _, components_second = build_splits(
            rows,
            seed=17,
            heldout_fraction=0.2,
            train_fraction=0.7,
            val_fraction=0.15,
            min_heldout_label_rows=3,
            min_remaining_category_rows=3,
        )

        self.assertEqual(components_first, components_second)
        self.assertEqual(
            {name: [item["id"] for item in values] for name, values in first.items()},
            {name: [item["id"] for item in values] for name, values in second.items()},
        )

        patient_sets = {
            name: {item["patient_group"] for item in values}
            for name, values in first.items()
        }
        names = list(patient_sets)
        for left_index, left in enumerate(names):
            for right in names[left_index + 1 :]:
                self.assertFalse(patient_sets[left] & patient_sets[right])

        train_pdds = {item["canonical_pdd"] for item in first["train"]}
        heldout_pdds = {
            item["canonical_pdd"] for item in first["test_pdd_heldout"]
        }
        self.assertFalse(train_pdds & heldout_pdds)
        seen_pdds = {
            item["canonical_pdd"]
            for split in ("train", "val_seen", "test_seen")
            for item in first[split]
        }
        self.assertTrue(seen_pdds <= train_pdds)

    def test_forbidden_pilot_label_excludes_its_connected_component(self):
        rows = []
        index = 0
        for pdd in "ABCDEFGH":
            for patient_index in range(6):
                index += 1
                patient = f"{pdd}-p{patient_index}"
                if patient_index == 0 and pdd in {"A", "B"}:
                    patient = "shared-a-b"
                rows.append(row(index, patient=patient, pdd=pdd, category="category"))

        splits, _, _ = build_splits(
            rows,
            seed=17,
            heldout_fraction=0.25,
            train_fraction=0.7,
            val_fraction=0.15,
            min_heldout_label_rows=3,
            min_remaining_category_rows=3,
            forbidden_heldout_pdds={"A"},
        )
        heldout = {item["canonical_pdd"] for item in splits["test_pdd_heldout"]}
        self.assertFalse({"A", "B"} & heldout)


if __name__ == "__main__":
    unittest.main()
