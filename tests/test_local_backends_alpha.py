# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import io

from PIL import Image

from simfoundry.models.local_backends import _drop_opaque_alpha

W, H = 1184, 864


def _png(image):
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def _decode(data):
    return Image.open(io.BytesIO(data))


def _frame(alpha, rgb=(200, 120, 40)):
    """RGBA frame of one colour, with `alpha` (an 'L' image of the same size) as its alpha channel."""
    image = Image.new("RGBA", alpha.size, rgb + (255,))
    image.putalpha(alpha)
    return image


def test_non_rgba_input_is_returned_unchanged():
    for mode in ("RGB", "L", "LA"):
        data = _png(Image.new(mode, (16, 16)))
        assert _drop_opaque_alpha(data) == data


def test_fully_opaque_rgba_becomes_rgb_with_same_pixels():
    frame = _frame(Image.new("L", (W, H), 255))
    out = _decode(_drop_opaque_alpha(_png(frame)))
    assert out.mode == "RGB"
    assert out.size == (W, H)
    assert out.tobytes() == frame.convert("RGB").tobytes()


def test_single_pixel_at_alpha_249_drops_alpha():
    alpha = Image.new("L", (W, H), 255)
    alpha.putpixel((350, 171), 249)
    assert _decode(_drop_opaque_alpha(_png(_frame(alpha)))).mode == "RGB"


def test_near_opaque_alpha_between_250_and_254_drops_alpha():
    # Every 1000th-pixel block has 98 pixels at 250-254 and the rest at 255: about 9.8%, none below 128.
    alpha = Image.new("L", (W, H), 255)
    alpha.putdata([250 + i % 5 if i % 1000 < 98 else 255 for i in range(W * H)])
    hist = alpha.histogram()
    assert 0.09 < sum(hist[250:255]) / (W * H) < 0.10
    assert sum(hist[:128]) == 0
    assert _decode(_drop_opaque_alpha(_png(_frame(alpha)))).mode == "RGB"


def test_cutout_with_30_percent_transparent_keeps_rgba():
    alpha = Image.new("L", (W, H), 255)
    alpha.putdata([0 if i % 10 < 3 else 255 for i in range(W * H)])
    data = _png(_frame(alpha))
    out = _drop_opaque_alpha(data)
    assert out == data
    assert _decode(out).getchannel("A").tobytes() == alpha.tobytes()


def test_cutout_with_1_percent_transparent_keeps_rgba():
    alpha = Image.new("L", (W, H), 255)
    alpha.putdata([0 if i % 100 == 0 else 255 for i in range(W * H)])
    data = _png(_frame(alpha))
    assert _drop_opaque_alpha(data) == data


def test_half_percent_threshold_is_inclusive_at_200_of_40000_pixels():
    # 0.5% of 200x200 is exactly 200 pixels, so 199 transparent pixels is not a cutout and 200 is.
    size = (200, 200)
    below = Image.new("L", size, 255)
    below.putdata([0] * 199 + [255] * (40000 - 199))
    assert _decode(_drop_opaque_alpha(_png(_frame(below)))).mode == "RGB"

    at = Image.new("L", size, 255)
    at.putdata([0] * 200 + [255] * (40000 - 200))
    data = _png(_frame(at))
    assert _drop_opaque_alpha(data) == data


def test_alpha_249_pixel_keeps_its_rgb_value():
    # Compositing over a background would change this colour; dropping the alpha channel must not.
    frame = _frame(Image.new("L", (W, H), 255))
    frame.putpixel((350, 171), (10, 20, 30, 249))
    out = _decode(_drop_opaque_alpha(_png(frame)))
    assert out.mode == "RGB"
    assert out.getpixel((350, 171)) == (10, 20, 30)
    assert out.tobytes() == frame.convert("RGB").tobytes()


def test_rgb_output_is_a_png():
    out = _drop_opaque_alpha(_png(_frame(Image.new("L", (W, H), 255))))
    assert _decode(out).format == "PNG"
