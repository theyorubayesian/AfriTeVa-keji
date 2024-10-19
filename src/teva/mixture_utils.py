from __future__ import annotations
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Optional

import gin
import seqio
from t5.data.utils import rate_num_examples

from teva.teva_tasks import TevaTasks


@gin.register
@dataclass
class MixtureRateConfig:
    scale: float = 1.0
    temperature: float = 1.0
    maximum: Optional[float] = None


def get_rate(
    mixture_or_task: str | seqio.Task | seqio.Mixture,
    scale: float = 1.0,
    temperature: float = 1.0,
    maximum: int | float | None  = None,
) -> callable | float:
    if isinstance(mixture_or_task, seqio.Task) or \
        mixture_or_task in seqio.TaskRegistry.names():

        return partial(
            rate_num_examples,
            scale=scale,
            temperature = temperature,
            maximum=maximum
        )

    return rate_num_examples_for_mixtures(
        task=mixture_or_task,
        scale=scale,
        temperature=temperature,
        maximum=maximum
    )


def rate_num_examples_for_mixtures(
    task: str | seqio.Mixture,
    maximum: Optional[int] = None,
    scale: float = 1.0,
    temperature: float = 1.0,
    fallback_to_num_input_examples: bool = True,
    split: str = "train",
) -> float:
    ret = 0
    if isinstance(task, str):
        task = seqio.MixtureRegistry.get(task)
    
    for t in task.tasks:
        try:
            if t.cache_dir and not fallback_to_num_input_examples:
                ret += t.get_cached_stats(split)["examples"]
            else:
                ret += t.num_input_examples(split) 
        except (ValueError, KeyError):
            # Some tasks may not have a train split
            continue

    ret *= scale
    if maximum:
        if isinstance(maximum, float):
            maximum *= ret
        ret = min(ret, maximum)
    if temperature != 1.0:
        ret = ret ** (1.0 / temperature)
    return ret
