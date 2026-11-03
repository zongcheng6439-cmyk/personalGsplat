"""Check opacity entropy gradients, visibility, schedule and PGSR composition.

From repository root in the training environment:
python tests/test_opacity_entropy.py
Inputs are synthetic logits/radii; no scene data or training outputs are written.
"""
import ast
import importlib.util
import math
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
# Load the pure torch losses without importing gsplat's CUDA package initializer.
spec = importlib.util.spec_from_file_location("opacity_losses", ROOT / "examples/opacity_regularization.py")
losses = importlib.util.module_from_spec(spec)
spec.loader.exec_module(losses)
opacity_entropy_loss = losses.opacity_entropy_loss

# Exercise the actual hook/schedule source without constructing a renderer.
# These tests cover the changed code; they are not an end-to-end training run.
source = ast.parse((ROOT / "examples/train_voxel_plane_mcmc.py").read_text())
runner_source = next(n for n in source.body if isinstance(n, ast.ClassDef) and n.name == "Runner")
config_source = next(n for n in source.body if isinstance(n, ast.ClassDef) and n.name == "Config")
hooks = [n for n in runner_source.body if isinstance(n, ast.FunctionDef)
         and n.name in ("extra_training_loss", "_opacity_entropy_loss")]
adjust = next(n for n in config_source.body if isinstance(n, ast.FunctionDef) and n.name == "adjust_steps")
fields = [n for n in config_source.body if isinstance(n, ast.AnnAssign)
          and (n.target.id.startswith("opacity_entropy_") or n.target.id in ("pgsr_start_iter", "pgsr_warmup_steps"))]
module = ast.Module(body=[
    ast.ClassDef(name="Runner", bases=[], keywords=[], body=hooks, decorator_list=[]),
    ast.ClassDef(name="Config", bases=[ast.Name(id="BaseConfig", ctx=ast.Load())], keywords=[],
                 body=fields+[adjust], decorator_list=[]),
], type_ignores=[])
class BaseConfig:
    def __init__(self, **kwargs): self.__dict__.update(kwargs)
    def adjust_steps(self, factor): pass
namespace = dict(torch=torch, opacity_entropy_loss=opacity_entropy_loss, BaseConfig=BaseConfig)
exec(compile(ast.fix_missing_locations(module), "trainer_hook_under_test", "exec"), namespace)
Runner, Config = namespace['Runner'], namespace['Config']


class OpacityEntropyTests(unittest.TestCase):
    def runner(self, **kwargs):
        r = Runner.__new__(Runner)
        settings = dict(opacity_entropy_lambda=.02, opacity_entropy_start_iter=10,
                        opacity_entropy_warmup_steps=10, opacity_entropy_stop_iter=30,
                        tb_every=0)
        settings.update(kwargs)
        r.cfg = SimpleNamespace(**settings)
        r.world_rank = 0
        r.splats = {'opacities': torch.tensor([-2., 2., 1.], requires_grad=True)}
        r._pgsr_extra_loss = lambda **context: None
        return r

    def test_formula_and_gradient(self):
        z = torch.tensor([-8., -2., 0., 2., 8.], dtype=torch.float64, requires_grad=True)
        p = z.sigmoid()
        expected = -(p * p.log() + (1-p) * (1-p).log()).mean()
        actual = opacity_entropy_loss(z)
        torch.testing.assert_close(actual, expected)
        actual.backward()
        torch.testing.assert_close(z.grad, -z.detach()*p.detach()*(1-p.detach())/z.numel())
        self.assertGreater(z.grad[1].item(), 0)  # gradient descent reduces negative logits
        self.assertLess(z.grad[3].item(), 0)     # and increases positive logits
        self.assertAlmostEqual(opacity_entropy_loss(torch.zeros(4)).item(), math.log(2), places=6)
        self.assertTrue(torch.autograd.gradcheck(opacity_entropy_loss, (z.detach().requires_grad_(True),)))

    def test_extreme_empty_half_and_cuda(self):
        devices = ['cpu'] + (['cuda'] if torch.cuda.is_available() else [])
        for device in devices:
            for dtype in (torch.float32, torch.float16):
                z = torch.tensor([-1000., -100., 0., 100., 1000.], dtype=dtype, device=device, requires_grad=True)
                loss = opacity_entropy_loss(z)
                loss.backward()
                self.assertTrue(torch.isfinite(loss).item())
                self.assertTrue(torch.isfinite(z.grad).all().item())
            empty = torch.empty(0, device=device, requires_grad=True)
            loss = opacity_entropy_loss(empty)
            loss.backward()
            self.assertEqual(loss.item(), 0)
        print('Tested devices:', devices)

    def test_schedule_visibility_and_pgsr(self):
        r = self.runner()
        # Gaussian 0 visible in view 0, Gaussian 1 in view 1; 2 never visible.
        info = {'radii': torch.tensor([[[1,1],[0,0],[0,0]], [[0,0],[2,2],[0,0]]])}
        self.assertIsNone(r.extra_training_loss(info, step=9))
        self.assertEqual(r.extra_training_loss(info, step=10).item(), 0)
        expected = opacity_entropy_loss(r.splats['opacities'][:2])
        torch.testing.assert_close(r.extra_training_loss(info, step=15), .01 * expected)
        loss = r.extra_training_loss(info, step=20)
        torch.testing.assert_close(loss, .02 * expected)
        loss.backward()
        self.assertEqual(r.splats['opacities'].grad[2].item(), 0)
        self.assertIsNone(r.extra_training_loss(info, step=30))
        r._pgsr_extra_loss = lambda **context: expected * 3
        torch.testing.assert_close(r.extra_training_loss(info, step=20), 3.02 * expected)
        r.cfg.opacity_entropy_lambda = 0
        torch.testing.assert_close(r.extra_training_loss({}, step=20), 3 * expected)
        r.cfg.opacity_entropy_lambda = .02
        r._pgsr_extra_loss = lambda **context: None
        self.assertEqual(r.extra_training_loss({'radii': torch.zeros(1,3,2)}, step=20).item(), 0)
        r.cfg.opacity_entropy_warmup_steps = 0
        r.cfg.opacity_entropy_stop_iter = -1
        torch.testing.assert_close(r.extra_training_loss({'radii': torch.tensor([[1,1,0]])}, step=100), .02 * expected)

    def test_logging_and_config_scaling(self):
        r = self.runner(tb_every=1)
        values = {}
        r.writer = SimpleNamespace(add_scalar=lambda name,value,step: values.update({name:value}))
        r.extra_training_loss({'radii': torch.ones(1,3,2)}, step=20)
        self.assertEqual(values['train/opacity_entropy_visible_count'], 3)
        self.assertIn('train/opacity_entropy_weighted', values)
        for factor in (0.5, 2):
            cfg = Config(opacity_entropy_start_iter=10, opacity_entropy_warmup_steps=20,
                      opacity_entropy_stop_iter=50)
            cfg.adjust_steps(factor)
            self.assertEqual((cfg.opacity_entropy_start_iter,cfg.opacity_entropy_warmup_steps,cfg.opacity_entropy_stop_iter), (int(10*factor),int(20*factor),int(50*factor)))
            self.assertEqual(cfg.opacity_entropy_lambda, 0)


if __name__ == '__main__':
    unittest.main()
