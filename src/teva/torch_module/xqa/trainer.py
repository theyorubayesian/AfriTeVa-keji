import math
import time
from typing import Optional

from datasets import Dataset
from transformers import Seq2SeqTrainer
from transformers.trainer_utils import PredictionOutput, speed_metrics


class xQATrainer(Seq2SeqTrainer):
    def evaluate(
        self,
        eval_dataset: Optional[Dataset] = None,
        ignore_keys: Optional[list[str]] = None,
        metric_key_prefix: str = "eval",
        **gen_kwargs,
    ) -> dict[str, float]:
        generation_kwargs = gen_kwargs.copy()
        generation_kwargs["max_length"] = gen_kwargs.get("max_length") or self.args.generation_max_length
        generation_kwargs["num_beams"] = gen_kwargs.get("num_beams") or self.args.generation_num_beams

        eval_dataset = eval_dataset or self.eval_dataset
        eval_dataloader = self.get_eval_dataloader(eval_dataset)

        compute_metrics = self.compute_metrics
        self.compute_metrics = None

        start_time = time.time()
        eval_loop = self.prediction_loop if self.args.use_legacy_prediction_loop else self.evaluation_loop

        try:
            output = eval_loop(
                eval_dataloader,
                description="Evaluation",
                prediction_loss_only=True if compute_metrics is None else None,
                ignore_keys=ignore_keys,
                metric_key_prefix=metric_key_prefix
            )
        finally:
            self.compute_metrics = compute_metrics

        total_batch_size = self.args.eval_batch_size * self.args.world_size

        if f"{metric_key_prefix}_jit_compilation_time" in output.metrics:
            start_time += output.metrics[f"{metric_key_prefix}_jit_compilation_time"]
        
        output.metrics.update(
            speed_metrics(
                metric_key_prefix,
                start_time,
                num_samples=output.num_samples,
                num_steps=math.ceil(output.num_samples / total_batch_size),
            )
        )

        # if self
