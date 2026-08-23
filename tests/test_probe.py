from __future__ import annotations

import unittest

from mlx_gptq_calibration.campaign import CampaignError
from mlx_gptq_calibration.cli import _require_satisfied


class ProbeTests(unittest.TestCase):
    def test_rejects_underprovisioned_host(self) -> None:
        probe = {
            "gpus": [{"name": "RTX 5070 Laptop GPU", "memory_mib": 8151}],
            "requirements": {
                "minimum_gpu_count": 4,
                "minimum_vram_mib": 20000,
                "satisfied": False,
            },
        }
        with self.assertRaisesRegex(CampaignError, "underprovisioned"):
            _require_satisfied(probe, allow=False)

    def test_override_is_explicit(self) -> None:
        probe = {
            "gpus": [],
            "requirements": {
                "minimum_gpu_count": 4,
                "minimum_vram_mib": 20000,
                "satisfied": False,
            },
        }
        _require_satisfied(probe, allow=True)


if __name__ == "__main__":
    unittest.main()
