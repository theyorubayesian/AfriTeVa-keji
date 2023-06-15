from typing import Dict, List, NewType, TypedDict, Union

import tensorflow as tf

Language = NewType('Language', str)


class DatasetStatistics(TypedDict, total=False):
    dev: int
    eval: int
    train: int
    test: int
    val: int
    validation: int

class CorpusStatistics(TypedDict):
    language: DatasetStatistics


class SplitStatistics(TypedDict, total=False):
    dev: Dict[Language, int]
    dev: Dict[Language, int]
    eval: Dict[Language, int]
    train: Dict[Language, int]
    test: Dict[Language, int]
    val: Dict[Language, int]
    validation: Dict[Language, int]


def get_labels(labels_file: str) -> List[str]:
    with tf.io.gfile.GFile(labels_file) as f:
        return f.read().splitlines()


def get_dataset_statistics(file: str) -> Union[CorpusStatistics, SplitStatistics]:
    stats = {}

    with tf.io.gfile.GFile(file) as f:
        for line in f:
            num_input_examples, language_and_split = line.split()
            split, file = language_and_split.split("/")
            language = file.split(".")[0]

            if split in ["dev", "eval", "val", "validation"]:
                split = "validation"
            
            if language in stats:
                stats[language][split] = int(num_input_examples)
            else:
                stats[language] = {split: int(num_input_examples)}
    
    return stats
