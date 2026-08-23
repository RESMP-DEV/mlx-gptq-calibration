from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from mlx_gptq_calibration.campaign import CampaignError, load_campaign
from mlx_gptq_calibration.cli import DEFAULT_CAMPAIGN, stage_a_argv


class CampaignTests(unittest.TestCase):
    def test_default_campaign_is_valid_and_pinned(self) -> None:
        campaign = load_campaign(DEFAULT_CAMPAIGN)
        self.assertEqual(campaign.data["model"]["id"], "Qwen/Qwen3.8-27B")
        self.assertEqual(campaign.data["dataset"]["id"], "RESMP-DEV/ptq-calibration-corpus")
        self.assertEqual(campaign.data["calibration"]["mode"], "mxfp4")
        self.assertEqual(campaign.data["calibration"]["group_size"], 32)

    def test_stage_a_command_uses_raw_corpus_and_language_layers(self) -> None:
        campaign = load_campaign(DEFAULT_CAMPAIGN)
        argv = stage_a_argv(campaign, 4)
        self.assertIn("cuda:0,cuda:1,cuda:2,cuda:3", argv)
        self.assertIn("model.language_model.layers", argv)
        self.assertIn(
            ".local/share/mlx-gptq-calibration/qwen3.8-27b-mxfp4/inputs/calibration.txt",
            argv,
        )
        self.assertNotIn("calibration_256x2048.npy", argv)

    def test_rejects_invalid_native_mxfp_shape(self) -> None:
        data = json.loads(DEFAULT_CAMPAIGN.read_text())
        data["calibration"]["group_size"] = 64
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.json"
            path.write_text(json.dumps(data))
            with self.assertRaisesRegex(CampaignError, "mxfp4 requires"):
                load_campaign(path)


if __name__ == "__main__":
    unittest.main()
