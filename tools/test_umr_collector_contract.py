#!/usr/bin/env python3
"""Tests for the formal UMR collection-time code fingerprint."""

import os
import sys
import unittest


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from tools.umr_collector_contract import (  # noqa: E402
    COLLECTOR_CODE_PATHS,
    collector_code_fingerprint,
    current_collector_contract,
)


class UMRCollectorContractTest(unittest.TestCase):

    def test_current_contract_is_complete_and_deterministic(self):
        first = current_collector_contract(REPO_ROOT)
        second = current_collector_contract(REPO_ROOT)

        self.assertEqual(first, second)
        self.assertEqual(first["format"], "umr_collector_code_v1")
        self.assertEqual(
            [record["path"] for record in first["records"]],
            list(COLLECTOR_CODE_PATHS))
        self.assertIn("tools/dist_test.sh", COLLECTOR_CODE_PATHS)
        self.assertEqual(len(first["fingerprint"]), 64)
        for record in first["records"]:
            self.assertEqual(len(record["sha256"]), 64)
            self.assertTrue(os.path.isfile(
                os.path.join(REPO_ROOT, record["path"])))

    def test_any_record_hash_change_invalidates_fingerprint(self):
        contract = current_collector_contract(REPO_ROOT)
        changed = [dict(record) for record in contract["records"]]
        changed[0]["sha256"] = "0" * 64

        self.assertNotEqual(
            contract["fingerprint"], collector_code_fingerprint(changed))


if __name__ == "__main__":
    unittest.main()
