"""FastWan workload and admission regressions; no generation."""

import copy
import tempfile
import unittest
from pathlib import Path

from common import HERE, read_json
from fastwan import geometry, tensor_headers, validate_profile
from runtime import make_request, verify_attention


class FastWanTests(unittest.TestCase):
    def test_wan_backend_check_uses_bound_impl_without_h3_private_enum(self):
        from types import SimpleNamespace

        impl_type = type("VideoSparseAttentionImpl", (), {})
        block_type = type("WanTransformerBlock_VSA", (), {})
        block = block_type()
        block.attn1 = SimpleNamespace(attn_impl=impl_type())
        model = SimpleNamespace(blocks=[block])
        self.assertEqual(verify_attention(model), ["VideoSparseAttentionImpl"])
        block.attn1.attn_impl = object()
        with self.assertRaisesRegex(ValueError, "Unexpected executed attention"):
            verify_attention(model)

    def test_workload_and_native_request(self):
        p = read_json(HERE / "configs/fastwan21_14b.json")
        validate_profile(p)
        self.assertEqual(geometry(p["sampling"]), ([21, 45, 80], 1440, 144))
        request = make_request(p, Path("/tmp/test.mp4"))
        for key in [
            "height",
            "width",
            "num_frames",
            "fps",
            "seed",
            "guidance_scale",
            "num_inference_steps",
        ]:
            self.assertEqual(request[key], p["sampling"][key])
        self.assertEqual(request["flow_shift"], 8.0)
        self.assertNotIn("audio_flow_shift", request)
        self.assertNotIn("target", request)

    def test_reject_changed_compute_or_topology(self):
        p = read_json(HERE / "configs/fastwan21_14b.json")
        for key, value in [
            ("num_inference_steps", 4),
            ("guidance_scale", 5),
            ("width", 1279),
        ]:
            changed = copy.deepcopy(p)
            changed["sampling"][key] = value
            with self.assertRaises(ValueError):
                validate_profile(changed)
        changed = copy.deepcopy(p)
        changed["server"]["sp_degree"] = 2
        with self.assertRaises(ValueError):
            validate_profile(changed)

    def test_reject_truncated_checkpoint_payload(self):
        import json
        import struct

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "transformer").mkdir()
            header = json.dumps(
                {"gate": {"dtype": "BF16", "shape": [2], "data_offsets": [0, 4]}}
            ).encode()
            shard = root / "transformer/model.safetensors"
            shard.write_bytes(struct.pack("<Q", len(header)) + header + b"\0\0")
            with self.assertRaisesRegex(ValueError, "truncated"):
                tensor_headers(root)


if __name__ == "__main__":
    unittest.main()
