"""Verify the raw source and deterministic production cleaning output."""

import hashlib
import unittest

from src.cleaning.pipeline import RAW_DATA_PATH, clean_and_prepare_dataset
from src.models.common.features import FEATURES


EXPECTED_RAW_SHA256 = "82e4b2483fbd7919b4572cbefa7658fcd9d193fef3a5a1d835a9496b07017e77"


class CleaningPipelineTests(unittest.TestCase):
    """Check source integrity, shape, schema, and validation."""

    def test_production_cleaning(self) -> None:
        raw_hash = hashlib.sha256(RAW_DATA_PATH.read_bytes()).hexdigest()
        self.assertEqual(raw_hash, EXPECTED_RAW_SHA256)
        raw, prepared, validation = clean_and_prepare_dataset()
        self.assertEqual(raw.shape, (4000, 32))
        self.assertEqual(prepared.shape, (3791, 22))
        self.assertEqual(list(prepared.columns), ["price", *FEATURES])
        self.assertTrue(validation["prepared"]["valid"])


if __name__ == "__main__":
    unittest.main()

