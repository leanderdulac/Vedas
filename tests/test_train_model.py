"""TrainingArguments compatível com transformers 4.x e 5.x (sem download/treino)."""

from __future__ import annotations

import unittest
from pathlib import Path

from vedic_pipeline.train.model import (
    build_training_arguments,
    causal_lm_training_kwargs,
    filter_supported_kwargs,
)


class _TrainingArgsV5:
    """Espelha transformers>=5: sem overwrite_output_dir."""

    def __init__(
        self,
        output_dir: str,
        num_train_epochs: int = 1,
        per_device_train_batch_size: int = 2,
        learning_rate: float = 5e-5,
        weight_decay: float = 0.01,
        logging_steps: int = 10,
        save_steps: int = 500,
        save_total_limit: int = 2,
        prediction_loss_only: bool = True,
        fp16: bool = False,
        report_to=None,
        max_steps: int = -1,
        dataloader_drop_last: bool = False,
        remove_unused_columns: bool = False,
    ):
        self.output_dir = output_dir
        self.num_train_epochs = num_train_epochs
        self.per_device_train_batch_size = per_device_train_batch_size
        self.learning_rate = learning_rate
        self.max_steps = max_steps
        self.fp16 = fp16
        self.report_to = report_to


class _TrainingArgsStrict:
    """Recusa overwrite_output_dir no corpo (TypeError real do HF 5.x)."""

    def __init__(self, output_dir: str, max_steps: int = -1, **kwargs):
        if "overwrite_output_dir" in kwargs:
            raise TypeError(
                "TrainingArguments.__init__() got an unexpected keyword argument "
                "'overwrite_output_dir'"
            )
        self.output_dir = output_dir
        self.max_steps = max_steps
        self.extra = kwargs


class TrainModelArgsTests(unittest.TestCase):
    def test_filter_drops_overwrite_output_dir_when_unsupported(self):
        wanted = {
            "output_dir": "artifacts/model/checkpoints",
            "overwrite_output_dir": True,
            "max_steps": 10,
            "per_device_train_batch_size": 1,
            "not_a_real_arg": 123,
        }
        filtered = filter_supported_kwargs(_TrainingArgsV5, wanted)
        self.assertEqual(filtered["output_dir"], "artifacts/model/checkpoints")
        self.assertEqual(filtered["max_steps"], 10)
        self.assertEqual(filtered["per_device_train_batch_size"], 1)
        self.assertNotIn("overwrite_output_dir", filtered)
        self.assertNotIn("not_a_real_arg", filtered)

    def test_causal_kwargs_keep_cli_contract(self):
        kwargs = causal_lm_training_kwargs(
            out_dir=Path("artifacts/model_smoke_test"),
            epochs=1,
            batch_size=1,
            learning_rate=5e-5,
            max_steps=10,
            fp16=False,
            use_cuda=False,
        )
        self.assertEqual(kwargs["output_dir"], "artifacts/model_smoke_test/checkpoints")
        self.assertEqual(kwargs["max_steps"], 10)
        self.assertEqual(kwargs["per_device_train_batch_size"], 1)
        self.assertFalse(kwargs["fp16"])
        self.assertIn("overwrite_output_dir", kwargs)

    def test_build_args_succeeds_on_v5_like_class(self):
        kwargs = causal_lm_training_kwargs(
            out_dir=Path("/tmp/vedas-model"),
            epochs=1,
            batch_size=1,
            learning_rate=5e-5,
            max_steps=10,
            fp16=True,
            use_cuda=False,
        )
        args = build_training_arguments(_TrainingArgsV5, kwargs)
        self.assertEqual(args.output_dir, "/tmp/vedas-model/checkpoints")
        self.assertEqual(args.max_steps, 10)
        self.assertEqual(args.per_device_train_batch_size, 1)
        self.assertFalse(args.fp16)
        self.assertFalse(hasattr(args, "overwrite_output_dir"))

    def test_build_args_retries_when_class_raises_unexpected_kwarg(self):
        class _NamedButRaises:
            calls = 0

            def __init__(
                self,
                output_dir: str,
                overwrite_output_dir: bool = True,
                max_steps: int = -1,
            ):
                type(self).calls += 1
                if type(self).calls == 1:
                    raise TypeError(
                        "TrainingArguments.__init__() got an unexpected keyword argument "
                        "'overwrite_output_dir'"
                    )
                self.output_dir = output_dir
                self.max_steps = max_steps

        args = build_training_arguments(
            _NamedButRaises,
            {
                "output_dir": "out/checkpoints",
                "overwrite_output_dir": True,
                "max_steps": 8,
            },
        )
        self.assertEqual(args.output_dir, "out/checkpoints")
        self.assertEqual(args.max_steps, 8)
        self.assertEqual(_NamedButRaises.calls, 2)

    def test_build_args_filters_then_constructs_kwargs_only_class(self):
        args = build_training_arguments(
            _TrainingArgsStrict,
            {
                "output_dir": "out/checkpoints",
                "overwrite_output_dir": True,
                "max_steps": 8,
            },
        )
        self.assertEqual(args.output_dir, "out/checkpoints")
        self.assertEqual(args.max_steps, 8)
        self.assertNotIn("overwrite_output_dir", args.extra)

    def test_installed_transformers_accepts_filtered_kwargs(self):
        try:
            from transformers import TrainingArguments
        except ImportError:
            self.skipTest("transformers não instalado")

        kwargs = causal_lm_training_kwargs(
            out_dir=Path("/tmp/vedas-ta-probe"),
            epochs=1,
            batch_size=1,
            learning_rate=5e-5,
            max_steps=10,
            fp16=False,
            use_cuda=False,
        )
        args = build_training_arguments(TrainingArguments, kwargs)
        self.assertTrue(str(args.output_dir).endswith("checkpoints"))
        self.assertEqual(args.max_steps, 10)
        self.assertEqual(args.per_device_train_batch_size, 1)


if __name__ == "__main__":
    unittest.main()
