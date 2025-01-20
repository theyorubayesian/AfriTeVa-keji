from typing import Iterator, Optional

import numpy as np
import seqio
import torch
import tensorflow as tf
from torch.utils.data import IterableDataset

from teva.tasks import setup_tasks
from teva.teva_tasks import TevaTasks


class SeqioDataset:
    class IterableSeqioDataset(IterableDataset):
        def __init__(self, iterator: tf.data.NumpyIterator):
            self.iterator = iterator
        
        def __iter__(self) -> Iterator[torch.Tensor]:
            for numpy_item in iter(self.iterator):
                yield self.tensorize(numpy_item)

        @staticmethod
        def tensorize(item: dict[str, np.ndarray | str | bytes]):
            for key, value in item.items():
                if isinstance(value, np.ndarray):
                    item[key] = torch.from_numpy(value.copy())
            return item
    
    def __init__(self, task: TevaTasks, splits: list[str] = None, gin_file: Optional[str] = None, **seqio_kwargs):
        self._iterator_dict = self._get_iterator(task, splits, gin_file, **seqio_kwargs)
        self._datasets = {}
    
    def get_dataset(self, split: str) -> IterableSeqioDataset:
        if split not in self._datasets:
            ds_iterator = self._iterator_dict[split]
            self._datasets[split] = self.IterableSeqioDataset(ds_iterator)
        return self._datasets[split]
    
    @property
    def splits(self):
        return list(self._iterator_dict.keys())
    
    @staticmethod
    def _get_iterator(task: TevaTasks, splits: list[str] = None, gin_file: Optional[str] = None, **seqio_kwargs) -> dict[str, tf.data.NumpyIterator]:
        if gin_file is not None:
            import gin  # pylint: disable=import-outside-toplevels
            gin.parse_config_file(gin_file)
            setup_tasks_using_gin = gin.configurable(setup_tasks)
            setup_tasks_using_gin([task])
        else:
            setup_tasks([task])

        for registry in (seqio.TaskRegistry, seqio.MixtureRegistry):
            try:
                task_or_mixture = registry.get(task.value)
                break
            except ValueError:
                pass
        else:
            raise ValueError(f"Task {task.value} not found in seqio registry")

        output_dataset = {}

        splits = splits or ("train", "validation", "test")
        for split in splits:
            try:
                output_dataset[split] = task_or_mixture.get_dataset(
                    split=split, **seqio_kwargs
                ).as_numpy_iterator()
            except ValueError:
                pass

        # TODO: @theyorubayesian - Log available splits for dataset

        return output_dataset
