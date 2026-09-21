"""CPU tests require only PyTorch; CUDA mixed precision is tested if available."""

import unittest

import torch

from navsim.agents.diffusiondrive.modules.last_token_selector import LASTTokenSelector


class LASTTokenSelectorTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(331)

    def test_shape_vote_count_and_positive_bounded_gate(self):
        x = torch.randn(2, 256, 64).transpose(1, 2)  # non-contiguous
        y, global_token, aux = LASTTokenSelector()(x)
        self.assertEqual(y.shape, x.shape)
        self.assertEqual(global_token.shape, (2, 256))
        self.assertEqual(aux["indices"].shape, (2, 16, 256))
        torch.testing.assert_close(aux["vote"].sum(1), torch.full((2,), 16.0))
        self.assertTrue((aux["gate"] >= 0.75).all())
        self.assertTrue((aux["gate"] <= 1.25).all())
        torch.testing.assert_close(y, x * aux["gate"][..., None])

    def test_zero_constant_and_single_token_are_finite(self):
        for x in (torch.zeros(2, 64, 256), torch.ones(2, 64, 256),
                  torch.randn(2, 1, 7), torch.ones(2, 4, 1)):
            with self.subTest(shape=x.shape):
                y, global_token, aux = LASTTokenSelector()(x)
                for value in (y, global_token, aux["stability"], aux["gate"]):
                    self.assertTrue(torch.isfinite(value).all())
                if x.shape[1] == 1:
                    torch.testing.assert_close(y, x, rtol=0, atol=0)

    def test_k1_and_full_selection(self):
        x = torch.randn(2, 64, 13)
        _, _, aux = LASTTokenSelector(topk_ratio=1 / 64)(x)
        self.assertEqual(aux["indices"].shape, (2, 1, 13))
        y, global_token, aux = LASTTokenSelector(topk_ratio=1)(x)
        torch.testing.assert_close(y, x, rtol=0, atol=0)
        torch.testing.assert_close(global_token, x.mean(1))
        torch.testing.assert_close(aux["vote"], torch.ones(2, 64))

    def test_fp16_bfloat16_and_cuda_autocast(self):
        devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
        for device in devices:
            for dtype in (torch.float16, torch.bfloat16):
                if device == "cuda" and dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
                    continue
                with self.subTest(device=device, dtype=dtype):
                    x = torch.randn(2, 64, 13, device=device, dtype=dtype)
                    selector = LASTTokenSelector().to(device)
                    with torch.autocast(device_type=device, enabled=device == "cuda", dtype=dtype):
                        y, global_token, aux = selector(x)
                    self.assertEqual(y.dtype, dtype)
                    self.assertEqual(global_token.dtype, dtype)
                    self.assertEqual(aux["stability"].dtype, torch.float32)
                    self.assertTrue(torch.isfinite(y).all())
                    self.assertTrue(torch.isfinite(aux["stability"]).all())

    def test_gradient_reaches_original_values_in_both_outputs(self):
        x = torch.randn(2, 16, 9, requires_grad=True)
        y, global_token, aux = LASTTokenSelector()(x)
        y.sum().backward(retain_graph=True)
        torch.testing.assert_close(x.grad, aux["gate"][..., None].expand_as(x))
        x.grad.zero_()
        global_token.sum().backward()
        expected = torch.zeros_like(x)
        expected.scatter_add_(1, aux["indices"], torch.full_like(aux["indices"],
                              1 / aux["indices"].shape[1], dtype=x.dtype))
        torch.testing.assert_close(x.grad, expected)

    def test_kernel_has_dc_peak_and_preserves_constant_channels(self):
        selector = LASTTokenSelector(pre_norm=False)
        for channels in (1, 7, 8, 256):
            kernel = selector.gaussian_kernel_1d(channels, "cpu").flatten()
            self.assertEqual(kernel.argmax().item(), channels // 2)
            x = torch.ones(1, 4, channels)
            spectrum = torch.fft.fftshift(torch.fft.fft(x, dim=-1), dim=-1)
            lowpass = torch.fft.ifft(torch.fft.ifftshift(spectrum * kernel, dim=-1), dim=-1).real
            torch.testing.assert_close(lowpass, x, rtol=1e-6, atol=1e-6)

    def test_signed_score_selects_positive_spatial_token(self):
        x = torch.tensor([1.0, 2.0, -3.0]).view(1, 3, 1).expand(1, 3, 8)
        _, global_token, aux = LASTTokenSelector(topk_ratio=1 / 3, pre_norm=False)(x)
        torch.testing.assert_close(aux["indices"], torch.ones(1, 1, 8, dtype=torch.long))
        torch.testing.assert_close(global_token, torch.full((1, 8), 2.0))

    def test_state_is_empty_and_alpha_zero_is_identity(self):
        selector = LASTTokenSelector(gate_alpha=0)
        self.assertEqual(dict(selector.state_dict()), {})
        self.assertEqual(list(selector.parameters()), [])
        x = torch.randn(1, 64, 256)
        torch.testing.assert_close(selector(x)[0], x, rtol=0, atol=0)

    def test_invalid_parameters(self):
        for kwargs in ({"topk_ratio": 0}, {"topk_ratio": 1.1}, {"sigma_scale": 0},
                       {"eps": 0}, {"gate_alpha": 1}, {"gate_alpha": -1},
                       {"sigma_scale": float("nan")}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                LASTTokenSelector(**kwargs)


if __name__ == "__main__":
    unittest.main()
