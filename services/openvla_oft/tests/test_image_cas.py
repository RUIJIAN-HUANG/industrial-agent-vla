from __future__ import annotations

import numpy as np

from openvla_oft.image_cas import ImageCas, ImageCasConfig


def test_image_cas_roundtrip_reads_verified_rgb(tmp_path):
    cas = ImageCas(
        ImageCasConfig(
            root=tmp_path / "cas",
            max_blob_bytes=16 * 1024 * 1024,
            max_pixels=4_194_304,
            cache_max_bytes=64 * 1024 * 1024,
            missing_retry_count=0,
            missing_retry_delay_ms=0,
        )
    )
    rgb = np.zeros((4, 5, 3), dtype=np.uint8)
    rgb[..., 2] = 255

    reference = cas.write_rgb(rgb, camera_id="CAM_B_TOP")
    resolved = cas.resolve_rgb(
        reference,
        expected_camera_id="CAM_B_TOP",
        expected_size=(5, 4),
    )

    assert resolved.width == 5
    assert resolved.height == 4
    assert resolved.camera_id == "CAM_B_TOP"
    assert resolved.image_sha256 == reference["image_sha256"]
    assert resolved.rgb.shape == (4, 5, 3)
    assert np.array_equal(resolved.rgb, rgb)
