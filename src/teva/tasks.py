import json
import os
from dataclasses import asdict
from functools import partial
from string import Template
from typing import Final, Literal, Optional, Sequence, Union

import gin
import seqio
import tensorflow as tf
from absl import logging
from dotenv import load_dotenv
from t5.data.preprocessors import span_corruption, summarize
from t5.data.utils import rate_num_examples
from t5.evaluation.metrics import accuracy, bleu, rouge, squad as squad_metrics

from teva.constants import *
from teva.metrics import chrf, weighted_multiclass_f1
from teva.mixture_utils import get_rate, MixtureRateConfig, rate_num_examples_for_mixtures
from teva.preprocessors import (
    afriqa,
    create_news_classification_example,
    jsonline_to_dict, 
    line_to_dict,
    squad,
    take_subset,
    translate
)
from teva.postprocessors import squad_postprocessor
from teva.teva_tasks import TevaTasks
from teva.utils import (
    get_dataset_statistics,
    get_labels,
    get_language_from_code,
    task_or_mix_exists,
    normalize,
    TaskNotFoundException,
)
from teva.vocab import DEFAULT_VOCAB

load_dotenv()

DATA_DIR=os.getenv("DATA_DIR", "data")

DEFAULT_OUTPUT_FEATURES: Final = {
    "inputs": seqio.Feature(vocabulary=DEFAULT_VOCAB, add_eos=True, required=False),
    "targets": seqio.Feature(vocabulary=DEFAULT_VOCAB, add_eos=True)
}


def add_wura_task():
    STATISTICS_PATH = Template(os.path.join(DATA_DIR, "wura/${corpus}-passages/stats"))
    DATASET_PATH = Template(os.path.join(DATA_DIR, "wura/${corpus}-passages/${split}/${language}.txt"))

    CORPORA: Final = ["wiki", "news"] 
    lm_tasks = []

    for corpus in CORPORA:
        corpus_tasks = []

        dataset_statistics = get_dataset_statistics(STATISTICS_PATH.substitute(corpus=corpus.capitalize()))
        for lang in WURA_LANGUAGES:
            if corpus == "news" and lang in ["arz"]:
                continue

            lang_config_name = f"{lang}_wura_{corpus}"

            seqio.TaskRegistry.add(
                lang_config_name,
                source=seqio.TextLineDataSource(
                    split_to_filepattern={
                        "train": DATASET_PATH.substitute(
                            corpus=corpus.capitalize(),
                            split="train",
                            language=lang
                        ),
                        "validation": DATASET_PATH.substitute(
                            corpus=corpus.capitalize(), 
                            split="eval",
                            language=lang
                        )
                    },
                    num_input_examples=dataset_statistics[lang]
                ),
                preprocessors=[
                    line_to_dict,
                    seqio.preprocessors.tokenize,
                    span_corruption,
                    seqio.preprocessors.append_eos_after_trim
                ],
                output_features=DEFAULT_OUTPUT_FEATURES,
                metric_fns=[]
            )

            corpus_tasks.append(lang_config_name)
        
        seqio.MixtureRegistry.add(corpus, corpus_tasks, default_rate=rate_num_examples)
        lm_tasks += corpus_tasks

    seqio.MixtureRegistry.add("wura", lm_tasks, default_rate=rate_num_examples)

# ---------------------------------------------------------------------------------
# Evaluation Tasks
# ---------------------------------------------------------------------------------

# --------------------------
# MasakhaNews Classification
# --------------------------
def add_masakhanews_task():
    MASAKHANEWS_DATASET_PATH = Template(os.path.join(DATA_DIR, "masakhanews/${language}/${split}.jsonl"))
    LABELS_PATH = Template(os.path.join(DATA_DIR, "masakhanews/${language}/labels.txt"))
    
    masakhanews_dataset_statistics = get_dataset_statistics(os.path.join(DATA_DIR, "masakhanews/stats"))
    # TODO: @theyorubayesian
    # This is a conversion from SplitStatistics to CorpusStatistics. Formalize it as a method.
    masakhanews_dataset_statistics = {
        language: {
            split: masakhanews_dataset_statistics[split][language]
            for split in ["train", "validation", "test"]
        } for language in MASAKHANEWS_LANGUAGES
    }

    masakhanews_tasks = []

    MASAKHANEWS_JSONLINE_SPECS = {
        field: tf.TensorSpec(tf.TensorShape([]), tf.string, name=field)
        for field in ["category", "headline", "text", "url"]
    }
    parse_masakhanews_jsonline = partial(jsonline_to_dict, specs=MASAKHANEWS_JSONLINE_SPECS)

    for language in MASAKHANEWS_LANGUAGES:
        lang_config_name = f"{language}_masakhanews"
        labels = get_labels(LABELS_PATH.substitute(language=language))
        
        weighted_f1 = weighted_multiclass_f1(labels)

        seqio.TaskRegistry.add(
            name=lang_config_name,
            source=seqio.TextLineDataSource(
                split_to_filepattern={
                        "train": MASAKHANEWS_DATASET_PATH.substitute(
                            split="train",
                            language=language
                        ),
                        "validation": MASAKHANEWS_DATASET_PATH.substitute(
                            split="dev",
                            language=language
                        ),
                        "test": MASAKHANEWS_DATASET_PATH.substitute(
                            split="test",
                            language=language
                        )
                    },
                num_input_examples=masakhanews_dataset_statistics[language]
            ),
            preprocessors=[
                parse_masakhanews_jsonline,
                create_news_classification_example,
                seqio.preprocessors.tokenize,
                seqio.preprocessors.append_eos_after_trim
            ],
            output_features=DEFAULT_OUTPUT_FEATURES,
            metric_fns=[accuracy, weighted_f1]
        )
        masakhanews_tasks.append(lang_config_name)

    seqio.MixtureRegistry.add("masakhanews", masakhanews_tasks, default_rate=rate_num_examples)

# -----------
# Translation
# -----------
# For inference: beam search - size: 5, length penalty: 0.6
def add_lafand_task():
    LAFAND_DATASET_PATH = Template(os.path.join(DATA_DIR, "lafand/${pivot}-${language}/${split}.json"))
    
    lafand_dataset_statistics = get_dataset_statistics(os.path.join(DATA_DIR, "lafand/stats"))
    lafand_dataset_statistics = {
        language: {
            split: lafand_dataset_statistics[split][language]
            for split in ["train", "validation", "test"]
            if language in lafand_dataset_statistics[split]     # Some languages do not have train
        } for language in LAFAND_LANGUAGES
    }

    lafand_en_xx_tasks = []
    lafand_xx_en_tasks = []

    for language in LAFAND_LANGUAGES:
        pivot = "en" if language in LAFAND_EN_PIVOT_LANGUAGES else "fr"

        lafand_en_xx_task_name = f"{pivot}_{language}_lafand_mt"
        en_xx_prefix = f"Translate {get_language_from_code(pivot)} to {get_language_from_code(language)}: "

        lafand_xx_en_task_name = f"{language}_{pivot}_lafand_mt"
        xx_en_prefix = f"Translate {get_language_from_code(language)} to {get_language_from_code(pivot)}: "

        JSONLINE_SPECS = {"translation": {
            _language: tf.TensorSpec(tf.TensorShape([]), tf.string, name=_language)
            for _language in [pivot, language]
        }}
        parse_lafand_jsonline = partial(jsonline_to_dict, specs=JSONLINE_SPECS, return_key="translation")
        
        lafand_source = seqio.TextLineDataSource(
            split_to_filepattern={
                "train": LAFAND_DATASET_PATH.substitute(
                    pivot=pivot,
                    split="train",
                    language=language
                ),
                "validation": LAFAND_DATASET_PATH.substitute(
                    pivot=pivot,
                    split="dev",
                    language=language
                ),
                "test": LAFAND_DATASET_PATH.substitute(
                    pivot=pivot,
                    split="test",
                    language=language
                )
            },
            num_input_examples=lafand_dataset_statistics[language]
        )

        seqio.TaskRegistry.add(
            name=lafand_en_xx_task_name,
            source=lafand_source,
            preprocessors=[
                parse_lafand_jsonline,
                partial(translate, prefix=en_xx_prefix, src_code=pivot, tgt_code=language),
                seqio.preprocessors.tokenize,
                seqio.preprocessors.append_eos_after_trim
            ],
            output_features=DEFAULT_OUTPUT_FEATURES,
            metric_fns=[bleu, chrf]
        )
        lafand_en_xx_tasks.append(lafand_en_xx_task_name)

        seqio.TaskRegistry.add(
            name=lafand_xx_en_task_name,
            source=lafand_source,
            preprocessors=[
                parse_lafand_jsonline,
                partial(translate, prefix=xx_en_prefix, src_code=language, tgt_code=pivot),
                seqio.preprocessors.tokenize,
                seqio.preprocessors.append_eos_after_trim
            ],
            output_features=DEFAULT_OUTPUT_FEATURES,
            metric_fns=[bleu, chrf]
        )
        lafand_xx_en_tasks.append(lafand_xx_en_task_name)

    seqio.MixtureRegistry.add("lafand_mt_en_xx", lafand_en_xx_tasks, default_rate=rate_num_examples)
    seqio.MixtureRegistry.add("lafand_mt_xx_en", lafand_xx_en_tasks, default_rate=rate_num_examples)
    seqio.MixtureRegistry.add("lafand_mt", [*lafand_xx_en_tasks, *lafand_en_xx_tasks], default_rate=rate_num_examples)

# -------------------
# XLSUM Summarization
# -------------------
# For inference: beam search - size: 4, length penalty: 0.6
# Batch size: 256
def add_xlsum_task(**mixture_rate_cfg_map: MixtureRateConfig) -> seqio.Mixture:
    if task_or_mix_exists("xlsum"):
        return seqio.MixtureRegistry.get("xlsum")

    XLSUM_DATASET_PATH = Template(os.path.join(DATA_DIR, "xlsum/${language}/${split}.json"))
    xlsum_dataset_statistics = get_dataset_statistics(os.path.join(DATA_DIR, "xlsum/stats"))
    
    xlsum_dataset_statistics = {
        language: {
            split: xlsum_dataset_statistics[split][language]
            for split in ["train", "validation", "test"]
        } for language in XLSUM_LANGUAGES
    }

    XLSUM_JSONLINE_SPECS = {
        field: tf.TensorSpec(tf.TensorShape([]), tf.string, name=field)
        for field in ["id", "url", "title", "summary", "text"]
    }
    parse_xlsum_jsonline = partial(jsonline_to_dict, specs=XLSUM_JSONLINE_SPECS)

    xlsum_tasks = []

    for language in XLSUM_LANGUAGES:
        xlsum_task_name = f"{language}_xlsum"

        if task_or_mix_exists(xlsum_task_name):
            xlsum_tasks.append(xlsum_task_name)
            continue

        seqio.TaskRegistry.add(
            xlsum_task_name,
            source=seqio.TextLineDataSource(
                split_to_filepattern={
                        "train": XLSUM_DATASET_PATH.substitute(
                            split="train",
                            language=language
                        ),
                        "validation": XLSUM_DATASET_PATH.substitute(
                            split="validation",
                            language=language
                        ),
                        "test": XLSUM_DATASET_PATH.substitute(
                            split="test",
                            language=language
                        )
                    },
                num_input_examples=xlsum_dataset_statistics[language]
            ),
            preprocessors=[
                parse_xlsum_jsonline,
                partial(summarize, article_key="text", summary_key="summary"),
                seqio.preprocessors.tokenize,
                seqio.preprocessors.append_eos_after_trim
            ],
            output_features=DEFAULT_OUTPUT_FEATURES,
            metric_fns=[rouge]
        )
        xlsum_tasks.append(xlsum_task_name)

    mixture = seqio.MixtureRegistry.add("xlsum", xlsum_tasks, default_rate=rate_num_examples)
    return mixture

# -------
# SQuADV2
# -------
def add_squad_task():
    SQUAD_DATASET_PATH = Template(os.path.join(DATA_DIR, "squad_v2/${split}.jsonl"))
    squad_dataset_statistics = {
        split: value["squad_v2"]
        for split, value in get_dataset_statistics(os.path.join(DATA_DIR, "squad_v2/stats")).items()
    }

    ANSWER_SPEC = {"answers": {
        "text": tf.TensorSpec(tf.TensorShape([None]), tf.string, name="text"),
        "answer_start": tf.TensorSpec(tf.TensorShape([None]), tf.int32, name="answer_start"),
    }}
    OTHER_SPECS = {
        field: tf.TensorSpec(tf.TensorShape([]), tf.string, name=field)
        for field in ["id", "title", "context", "question"]
    }
    SQUAD_SPECS = {**OTHER_SPECS, **ANSWER_SPEC}
    parse_squad_jsonline = partial(jsonline_to_dict, specs=SQUAD_SPECS)

    seqio.TaskRegistry.add(
        name="squad_v2",
        source=seqio.TextLineDataSource(
            split_to_filepattern={
                "train": SQUAD_DATASET_PATH.substitute(split="train"),
                "validation": SQUAD_DATASET_PATH.substitute(split="val"),
            },
            num_input_examples=squad_dataset_statistics
        ),
        preprocessors=[
            parse_squad_jsonline,
            squad,
            seqio.preprocessors.tokenize,
            seqio.preprocessors.append_eos_after_trim
        ],
        postprocess_fn=squad_postprocessor,
        output_features=DEFAULT_OUTPUT_FEATURES,
        metric_fns=[squad_metrics]
    )

# -------
# AfriQA
# -------
def add_afriqa_task(skip_train_split: bool = False):
    AFRIQA_DATASET_PATH = Template(
        os.path.join(
            DATA_DIR, 
            "afriqa/gold_passages/${language}/gold_span_passages.afriqa.${language}.${pivot}.${split}.json"
        )
    )

    afriqa_dataset_statistics = get_dataset_statistics(os.path.join(DATA_DIR, "afriqa/gold_passages/stats"))
    afriqa_dataset_statistics = {
        language: {
            split: afriqa_dataset_statistics[split][language]
            for split in ["train", "validation", "test"]
        } for language in AFRIQA_LANGUAGES
    }

    ANSWER_SPEC = {"answer_pivot": {
        "answer_start": tf.TensorSpec([None], tf.int32, name="answer_start"),
        "text": tf.TensorSpec([None], tf.string, name="text")
    }}
    OTHER_SPECS = {
        field: tf.TensorSpec([], tf.string, name=field) 
        for field in [
            "context", "id", "question_lang", 
            "question_translated", "title", "answer_lang"
        ]
    }
    AFRIQA_SPEC = {**OTHER_SPECS, **ANSWER_SPEC}
    parse_afriqa_jsonline = partial(jsonline_to_dict, specs=AFRIQA_SPEC)

    afriqa_tasks = []
    for language in AFRIQA_LANGUAGES:
        task_name = f"{language}_afriqa"
        pivot = "fr" if language in AFRIQA_FR_PIVOT_LANGUAGES else "en"
        
        split_to_filepattern={
            "train": AFRIQA_DATASET_PATH.substitute(split="train", language=language, pivot=pivot),
            "validation": AFRIQA_DATASET_PATH.substitute(split="dev", language=language, pivot=pivot),
            "test": AFRIQA_DATASET_PATH.substitute(split="test", language=language, pivot=pivot),
        }

        if skip_train_split:
            _ = split_to_filepattern.pop("train")

        seqio.TaskRegistry.add(
            name=task_name,
            source=seqio.TextLineDataSource(
                split_to_filepattern=split_to_filepattern,
                num_input_examples=afriqa_dataset_statistics[language]
            ),
            preprocessors=[
                parse_afriqa_jsonline,
                afriqa,
                seqio.preprocessors.tokenize,
                seqio.preprocessors.append_eos_after_trim
            ],
            postprocess_fn=squad_postprocessor,
            output_features=DEFAULT_OUTPUT_FEATURES,
            metric_fns=[squad_metrics]
        )
        afriqa_tasks.append(task_name)
    
    seqio.MixtureRegistry.add("afriqa", afriqa_tasks, default_rate=rate_num_examples)

def add_evaluation_tasks():
    ...

# ---
# Aya
# ---
@gin.register
def create_aya_dataset_mixture(
    languages: Sequence[str],
    dataset_name: str,
    suffix: Optional[str] = None,
    default_mixture_rate_cfg = MixtureRateConfig(),
    **mixture_rate_cfg_map: MixtureRateConfig
) -> Optional[seqio.Mixture]:
    prefix = [dataset_name, suffix][bool(suffix)]
    if task_or_mix_exists(f"{prefix}_aya"):
        return seqio.MixtureRegistry.get(f"{prefix}_aya")

    DATASET_PATH = Template(os.path.join(DATA_DIR, f"aya_african_subset/data/{dataset_name}", "${split}/${language}.jsonl"))

    AYA_DATASET_STATISTICS = get_dataset_statistics(os.path.join(DATA_DIR, "aya_african_subset/statistics.json"))
    
    TEXT_SPEC = {
        field: tf.TensorSpec([], tf.string, name=field) 
        for field in [
            'inputs', 'targets', 'dataset_name', 
            'sub_dataset_name', 'task_type', 'language', 
            'script', 'split'
        ]
    }

    ID_SPEC = {
        field: tf.TensorSpec([None], tf.int32, name=field)
        for field in ["id", "template_id"]
    }

    AYA_SPEC = {**TEXT_SPEC, **ID_SPEC}

    parse_aya_jsonline = partial(jsonline_to_dict, specs=AYA_SPEC)

    aya_tasks = []
    for language in languages:
        sources = {
            split: DATASET_PATH.substitute(split="train", language=language)
            for split in ["train", "validation", "test"]
            if tf.io.gfile.exists(DATASET_PATH.substitute(split=split, language=language))
        }
        if not sources:
            continue

        task_name = normalize(f"{language}_{prefix}_aya")

        if not task_or_mix_exists(task_name):
            seqio.TaskRegistry.add(
                name=task_name,
                source=seqio.TextLineDataSource(
                    split_to_filepattern=sources,
                    num_input_examples={
                        source: AYA_DATASET_STATISTICS[dataset_name][source][language]
                        for source in sources
                    }
                ),
                preprocessors=[
                    parse_aya_jsonline,
                    partial(take_subset, keys=["inputs", "targets", "dataset_name", "task_type", "language", "script"]),
                    seqio.preprocessors.tokenize,
                    seqio.preprocessors.append_eos_after_trim
                ],
                output_features=DEFAULT_OUTPUT_FEATURES,
                metric_fns=[]
            )

        mixture_rate_cfg = mixture_rate_cfg_map.get(
            f"{language}_mixture_cfg", default_mixture_rate_cfg)
        mixture_rate = get_rate(task_name, **asdict(mixture_rate_cfg))

        aya_tasks.append((task_name, mixture_rate))
    
    # if len(aya_tasks) > 1:
    mixture = seqio.MixtureRegistry.add(
        name=f"{prefix}_aya", 
        tasks=aya_tasks,
        default_rate=rate_num_examples
    )

    return mixture


@gin.register
def add_aya_human_task(
    languages: Sequence[str] = AYA_HUMAN_LANGUAGES,
    mixture_fn: callable = create_aya_dataset_mixture
) -> seqio.Mixture:
    assert AYA_HUMAN_LANGUAGES.issuperset(languages)
    aya_human_mixture = mixture_fn(
        languages, "aya-dataset", mixture_rate=rate_num_examples, suffix="human")
    return aya_human_mixture


@gin.register
def add_aya_translated_task(
    datasets: Sequence[str] = AYA_TEMPLATED_DATASETS,
    mixture_fn: callable = create_aya_dataset_mixture,
    **mixture_rate_cfg_map: MixtureRateConfig
) -> seqio.Mixture:
    assert AYA_TEMPLATED_DATASETS.issuperset(datasets)
    sub_mixtures = []
    
    for dataset_name in datasets:
        mixture = mixture_fn(
            AYA_LANGUAGES,
            dataset_name=dataset_name,
            mixture_rate=rate_num_examples
        )
        mixture_rate_cfg = mixture_rate_cfg_map.get(
            f"{normalize(dataset_name)}_mixture_cfg", MixtureRateConfig())
        
        sub_mixtures.append((mixture, get_rate(
            mixture, **asdict(mixture_rate_cfg))
        ))
    
    translated_aya = seqio.MixtureRegistry.add(
        "translated_aya", sub_mixtures
    )
    return translated_aya


@gin.register
def add_aya_templated_task(
    datasets: Sequence[str] = AYA_TEMPLATED_DATASETS,
    mixture_fn: callable = create_aya_dataset_mixture,
    **mixture_rate_cfg_map: MixtureRateConfig
) -> seqio.Mixture:
    assert AYA_TEMPLATED_DATASETS.issuperset(datasets)
    sub_mixtures = []
    
    for dataset_name in datasets:
        mixture = mixture_fn(
            AYA_LANGUAGES,
            dataset_name=dataset_name,
            mixture_rate=rate_num_examples
        )
        mixture_rate_cfg = mixture_rate_cfg_map.get(
            f"{normalize(dataset_name)}_mixture_cfg", MixtureRateConfig())
        
        sub_mixtures.append((mixture, get_rate(
            mixture, **asdict(mixture_rate_cfg))
        ))
    
    templated_aya = seqio.MixtureRegistry.add(
        "templated_aya", sub_mixtures
    )
    return templated_aya


@gin.register
def add_aya_collection_task(
    **mixture_rate_cfg_map: MixtureRateConfig
) -> seqio.Mixture:
    
    if task_or_mix_exists("aya_collection"):
        return seqio.MixtureRegistry.get("aya_collection")
    
    sub_mixtures = []

    for task in TevaTasks.get_aya_collection_tasks():
        task_name = default_task_factory[task]().name
        mixture_rate_cfg = mixture_rate_cfg_map.get(f"{task_name}_mixture_cfg", MixtureRateConfig())
        sub_mixtures.append((task_name, get_rate(task_name, **asdict(mixture_rate_cfg))))

    return seqio.MixtureRegistry.add("aya_collection", tasks=sub_mixtures)


# TODO: @theyorubayesian- pcm_Latn isn't included when downloading xP3x. Fix. 
@gin.register
def add_xp3x_task(
    languages: Sequence[str] = XP3X_LANGUAGE_CODES,
    **mixture_rate_cfg_map: MixtureRateConfig
) -> seqio.Mixture:
    if task_or_mix_exists("xP3x"):
        return seqio.MixtureRegistry.get("xP3x")
    
    assert XP3X_LANGUAGE_CODES.issuperset(languages)

    XP3X_DATASET_PATH = Template(os.path.join(DATA_DIR, "xP3x/${language}/*.jsonl"))
    xp3x_dataset_statistics = get_dataset_statistics(os.path.join(DATA_DIR, "xP3x/statistics.json"))

    XP3X_JSONLINE_SPECS = {
        field: tf.TensorSpec(tf.TensorShape([]), tf.string, name=field)
        for field in ["inputs", "language", "split", "template", "dataset", "config"]
    }
    parse_xp3x_jsonline = partial(jsonline_to_dict, specs=XP3X_JSONLINE_SPECS)

    xP3x_tasks = []

    for language in languages:
        task_name = f"{language}_xP3x"

        if task_or_mix_exists(task_name):
            continue

        xp3x_source = seqio.TextLineDataSource(
            split_to_filepattern={
                "train": XP3X_DATASET_PATH.substitute(language=language)
            },
            num_input_examples={"train" : xp3x_dataset_statistics[language]}
        )

        seqio.TaskRegistry.add(
            name=task_name,
            source=xp3x_source,
            preprocessors=[
                parse_xp3x_jsonline,
                partial(take_subset, keys=["inputs", "targets", "config"]),
                seqio.preprocessors.tokenize,
                seqio.preprocessors.append_eos_after_trim
            ],
            output_features=DEFAULT_OUTPUT_FEATURES,
            metric_fns=[]
        )

        mixture_rate_cfg = mixture_rate_cfg_map.get(
            f"{language}_mixture_cfg", MixtureRateConfig())
        mixture_rate = get_rate(task_name, **asdict(mixture_rate_cfg))

        xP3x_tasks.append((task_name, mixture_rate))
    
    mixture = seqio.MixtureRegistry.add(
        name="xP3x",
        tasks=xP3x_tasks,
        default_rate=rate_num_examples
    )
    return mixture


@gin.register
def add_octopack_osst(
    languages: Sequence[str] = OCTOPACK_LANGUAGES,
    **mixture_rate_cfg_map: MixtureRateConfig
) -> seqio.Mixture:
    if task_or_mix_exists("octopack_osst"):
        return seqio.TaskRegistry.get("octopack_osst")
    
    assert OCTOPACK_LANGUAGES.issuperset(languages)

    OCTOPACK_DATASET_PATH= Template(os.path.join(DATA_DIR, "octopack_osst", "oasst_octopack_${language}.jsonl"))
    octopack_dataset_statistics = get_dataset_statistics(os.path.join(DATA_DIR, "octopack_osst/statistics.json"))

    @tf.autograph.experimental.do_not_convert
    def read_file_fn(file):
        def _read_json(file):
            with tf.io.gfile.GFile(file) as f:
                lines = f.read().splitlines()

            for line in lines:
                data = json.loads(line)
                all_turns = []
                for turn in data["conversations"]:
                    all_turns.append(f"{turn['role'].capitalize()}: {turn['text']}")

                yield {
                    "inputs": "\n\n".join(all_turns[:-1]) + "\n\nAssistant: ",
                    "targets": all_turns[-1].replace("Assistant: ", "")
                }

        return tf.data.Dataset.from_generator(_read_json, args=(file,),
            output_signature={"inputs": tf.TensorSpec([], dtype=tf.string), "targets": tf.TensorSpec([], dtype=tf.string)}
        )

    language_tasks = []
    for language in languages:
        task_name = f"{language.replace('-', '_')}_octopack_osst"

        if task_or_mix_exists(task_name):
            continue

        octopack_source = seqio.FileDataSource(
            read_file_fn=read_file_fn,
            split_to_filepattern={"train": OCTOPACK_DATASET_PATH.substitute(language=language)},
            num_input_examples={"train": octopack_dataset_statistics[language]}
        )

        seqio.TaskRegistry.add(
            name=task_name,
            source=octopack_source,
            preprocessors=[
                seqio.preprocessors.tokenize,
                seqio.preprocessors.append_eos_after_trim
            ],
            output_features=DEFAULT_OUTPUT_FEATURES,
            metric_fns=[]
        )

        mixture_rate_cfg = mixture_rate_cfg_map.get(
            f"{language}_mixture_cfg", MixtureRateConfig())
        mixture_rate = get_rate(task_name, **asdict(mixture_rate_cfg))

        language_tasks.append((task_name, mixture_rate))

    mixture = seqio.MixtureRegistry.add(
        name="octopack_osst",
        tasks=language_tasks,
        default_rate=rate_num_examples
    )
    return mixture


def add_oig_small_chip2() -> seqio.Task:
    if task_or_mix_exists("oig_small_chip2"):
        return seqio.TaskRegistry.get("oig_small_chip2")
    
    OIG_DATASET_PATH = os.path.join(DATA_DIR, "OIG-small-chip2/train*.jsonl")
    oig_dataset_statistics = get_dataset_statistics(os.path.join(DATA_DIR, "OIG-small-chip2/statistics.json"))
    
    OIG_JSONLINE_SPECS = {
        field: tf.TensorSpec(tf.TensorShape([]), tf.string, name=field)
        for field in ["user", "chip2"]
    }
    parse_oig_jsonline = partial(jsonline_to_dict, specs=OIG_JSONLINE_SPECS)

    task = seqio.TaskRegistry.add(
        "oig_small_chip2",
        source=seqio.TextLineDataSource(
            split_to_filepattern={
                "train": OIG_DATASET_PATH
            },
            num_input_examples=oig_dataset_statistics
        ),
        preprocessors=[
            parse_oig_jsonline,
            partial(
                seqio.preprocessors.rekey,
                key_map={"inputs": "user", "targets": "chip2"}
            ),
            seqio.preprocessors.tokenize,
            seqio.preprocessors.append_eos_after_trim
        ],
        output_features=DEFAULT_OUTPUT_FEATURES,
        metric_fns=[]
    )
    return task


def add_tasksource_instruct() -> seqio.Task:
    if task_or_mix_exists("tasksource_instruct"):
        return seqio.TaskRegistry.get("tasksource_instruct")
    
    TASKSOURCE_DATASET_PATH = Template(os.path.join(DATA_DIR, "tasksource_instruct/${split}*.jsonl"))
    tasksource_dataset_statistics = get_dataset_statistics(os.path.join(DATA_DIR, "tasksource_instruct/statistics.json"))

    TASKSOURCE_JSONLINE_SPECS = {
        field: tf.TensorSpec(tf.TensorShape([]), tf.string, name=field)
        for field in ["inputs", "targets", "task"]
    }
    parse_tasksource_jsonline = partial(jsonline_to_dict, specs=TASKSOURCE_JSONLINE_SPECS)

    task = seqio.TaskRegistry.add(
        "tasksource_instruct",
        source=seqio.TextLineDataSource(
            split_to_filepattern={
                split: TASKSOURCE_DATASET_PATH.substitute(split=split)
                for split in ["train", "validation", "test"]
            },
            num_input_examples=tasksource_dataset_statistics
        ),
        preprocessors=[
            parse_tasksource_jsonline,
            seqio.preprocessors.tokenize,
            seqio.preprocessors.append_eos_after_trim
        ],
        output_features=DEFAULT_OUTPUT_FEATURES,
        metric_fns=[]
    )
    return task


def create_flan_collection_submix_task(
    dataset_name: FlanTask,
    flan_collection_statistics: FlanCollectionStatistics | None = None
) -> seqio.Mixture:
    if task_or_mix_exists(f"{dataset_name.value}_submix"):
        return seqio.MixtureRegistry.get(f"{dataset_name.value}_submix")
    
    FLAN_COLLECTION_DATASET_PATH = os.path.join(DATA_DIR, f"flan_collection/{dataset_name.value}/train*.jsonl")
    
    if flan_collection_statistics is None:
        flan_collection_statistics = get_dataset_statistics(os.path.join(DATA_DIR, "flan_collection/statistics.json"))
    
    submix_statistics = {"train": flan_collection_statistics[f"{dataset_name.value}_submix"]}
    TEXT_SPEC = {
        field: tf.TensorSpec([], tf.int32, name=field)
        for field in ["inputs", "targets", "task_source", "task_name", "template_type"]
    }

    parse_flan_jsonline = partial(jsonline_to_dict, specs=TEXT_SPEC)

    mixture = seqio.TaskRegistry.add(
        name=f"{dataset_name.value}_submix",
        source=seqio.TextLineDataSource(
            split_to_filepattern={"train": FLAN_COLLECTION_DATASET_PATH},
            num_input_examples=submix_statistics,
        ),
        preprocessors=[
            parse_flan_jsonline,
            partial(take_subset, keys=["inputs", "targets"]),
            seqio.preprocessors.tokenize,
            seqio.preprocessors.append_eos_after_trim
        ],
        output_features=DEFAULT_OUTPUT_FEATURES,
        metric_fns=[]
    )
    return mixture


def add_niv2_submix_filtered(flan_collection_statistics: FlanCollectionStatistics | None = None) -> seqio.Mixture:
    return create_flan_collection_submix_task(
        dataset_name=FlanTask.NIV2,
        flan_collection_statistics=flan_collection_statistics
    )


def add_flan2021_submix_filtered(flan_collection_statistics: FlanCollectionStatistics | None = None) -> seqio.Mixture:
    return create_flan_collection_submix_task(
        dataset_name=FlanTask.FLAN2021,
        flan_collection_statistics=flan_collection_statistics
    )


def add_cot_submix_filtered(flan_collection_statistics: FlanCollectionStatistics | None = None) -> seqio.Mixture:
    return create_flan_collection_submix_task(
        dataset_name=FlanTask.COT,
        flan_collection_statistics=flan_collection_statistics
    )


def add_t0_submix(flan_collection_statistics: FlanCollectionStatistics | None = None) -> seqio.Mixture:
    return create_flan_collection_submix_task(
        dataset_name=FlanTask.T0,
        flan_collection_statistics=flan_collection_statistics
    )


def add_dialog_submix(flan_collection_statistics: FlanCollectionStatistics | None = None) -> seqio.Mixture:
    return create_flan_collection_submix_task(
        dataset_name=FlanTask.DIALOG,
        flan_collection_statistics=flan_collection_statistics
    )


@gin.register
def add_flan_collection_task(**mixture_rate_cfg_map: MixtureRateConfig) -> seqio.Mixture:
    """
    This is a mixture of the following Flan tasks:
        * COT Submix
        * Dialog Submix
        * Flan 2021 Submix
        * NIV2 Submix
        * T0 Submix
    """
    if task_or_mix_exists("flan_collection"):
        return seqio.MixtureRegistry.get("flan_collection")
    
    flan_tasks = []

    for task in TevaTasks.get_flan_collection_tasks():
        task_name = default_task_factory[task]().name
        mixture_rate_cfg = mixture_rate_cfg_map.get(
            f"{task_name}_mixture_cfg", MixtureRateConfig())
        flan_tasks.append((task_name, get_rate(task_name, **asdict(mixture_rate_cfg))))
    
    mixture = seqio.MixtureRegistry.add(
        "flan_collection",
        tasks=flan_tasks
    )
    return mixture


@gin.register
def add_dpi_templated_tasks(**mixture_rate_cfg_map: MixtureRateConfig) -> seqio.Mixture:
    """
    This is a mixture of the following tasks/mixtures:
        * Octopack OSST
        * OpenInstruction Generalist
        * Filtered Flan Collection (NIV2, COT, FLAN2021, T0, Dialog)
        * TaskSource Instruct?
    """
    if task_or_mix_exists("dpi_templated"):
        return seqio.MixtureRegistry.get("dpi_templated")
    
    sub_mixtures = []

    for task in TevaTasks.get_dpi_templated_tasks(flatten_flan_collection=False):
        task_name = default_task_factory[task]().name
        mixture_rate_cfg = mixture_rate_cfg_map.get(
            f"{task_name}_mixture_cfg", MixtureRateConfig())
        sub_mixtures.append((task_name, get_rate(task_name, **asdict(mixture_rate_cfg))))
    
    return seqio.MixtureRegistry.add(
        "dpi_templated",
        tasks=sub_mixtures,
    )


@gin.register
def add_templated_instruction_ft_tasks(**mixture_rate_cfg_map: MixtureRateConfig) -> seqio.Mixture:
    """
    This is a mixture of the following mixtures:
        * Aya Templated
        * xP3x
        * Data Provenance Initiative
    """
    if task_or_mix_exists("templated_ift"):
        return seqio.MixtureRegistry.get("templated_ift")
    
    sub_mixtures = []

    for task in TevaTasks.get_templated_instruction_tasks(flatten_mixtures=False):
        task_name = default_task_factory[task]().name
        mixture_rate_cfg = mixture_rate_cfg_map.get(
            f"{task_name}_mixture_cfg", MixtureRateConfig())
        sub_mixtures.append((task_name, get_rate(task_name, **asdict(mixture_rate_cfg))))
    
    return seqio.MixtureRegistry.add(
        "templated_ift",
        tasks=sub_mixtures,
    )


@gin.register
def add_instruction_ft_tasks(
    human_mixture_cfg: MixtureRateConfig = MixtureRateConfig(),
    translated_mixture_cfg: MixtureRateConfig = MixtureRateConfig(),
    templated_mixture_cfg: MixtureRateConfig = MixtureRateConfig(),
) -> seqio.Mixture:
    """
    This is a mixture of the following mixtures:
        * TemplatedIFT
        * AyaHuman
        * AyaTranslated
    """
    if task_or_mix_exists("ift_mixture"):
        return seqio.MixtureRegistry.get("ift_mixture")
    
    aya_translated_mixture = default_task_factory[TevaTasks.TRANSLATED_AYA]()
    aya_human_mixture = default_task_factory[TevaTasks.HUMAN_AYA]()
    templated_ift_mixture = default_task_factory[TevaTasks.TEMPLATED_IFT]()

    return seqio.MixtureRegistry.add(
        "ift_mixture",
        [
            (aya_translated_mixture, get_rate(
                aya_translated_mixture, **asdict(translated_mixture_cfg))),
            (aya_human_mixture, get_rate(
                aya_human_mixture, **asdict(human_mixture_cfg))),
            (templated_ift_mixture, get_rate(
                templated_ift_mixture, **asdict(templated_mixture_cfg)))
        ]
    )


default_task_factory: dict[TevaTasks, callable] = {
    TevaTasks.WURA: add_wura_task,
    TevaTasks.EVAL: add_evaluation_tasks,
    TevaTasks.IFT: add_instruction_ft_tasks,
    # TevaTasks.SFT: add_supervised_ft_tasks,   # TODO: @theyorubayesian
    TevaTasks.MASAKHANEWS: add_masakhanews_task,
    TevaTasks.LAFAND: add_lafand_task,
    TevaTasks.XLSUM: add_xlsum_task,
    TevaTasks.SQUAD: add_squad_task,
    TevaTasks.AFRIQA: add_afriqa_task,
    TevaTasks.HUMAN_AYA: add_aya_human_task,
    TevaTasks.TEMPLATED_AYA: add_aya_templated_task,
    TevaTasks.TRANSLATED_AYA: add_aya_translated_task,
    TevaTasks.AYA_COLLECTION: add_aya_collection_task,
    TevaTasks.XP3X: add_xp3x_task,
    TevaTasks.FLAN2021_SUBMIX: add_flan2021_submix_filtered,
    TevaTasks.FLAN_COT_SUBMIX: add_cot_submix_filtered,
    TevaTasks.FLAN_DIALOG_SUBMIX: add_dialog_submix,
    TevaTasks.FLAN_NIV2_SUBMIX: add_niv2_submix_filtered,
    TevaTasks.FLAN_T0_SUBMIX: add_t0_submix,
    TevaTasks.FLAN_COLLECTION: add_flan_collection_task,
    TevaTasks.OIG_SMALL_CHIP2: add_oig_small_chip2,
    TevaTasks.OCTOPACK_OSST: add_octopack_osst,
    TevaTasks.TASKSOURCE_INSTRUCT: add_tasksource_instruct,
    TevaTasks.DPI_TEMPLATED: add_dpi_templated_tasks,
    TevaTasks.TEMPLATED_IFT: add_templated_instruction_ft_tasks
}


def get_task_func_from_factory(key: str | TevaTasks) -> callable:
    if isinstance(key, str):
        _key = TevaTasks[key.upper()]
        return default_task_factory[_key]
    return default_task_factory[key]


def setup_tasks(
    tasks: Sequence[TevaTasks] | Literal["all"],
    **kwargs: Union[callable, MixtureRateConfig],
):
    """
    kwargs may contain gin-configured task functions or MixtureRateConfig
    """
    configured_task_factory = {}
    for k, v in kwargs.items():
        if isinstance(k, str):
            try:
                k = TevaTasks[k.upper()]
            except KeyError:
                continue
    
        if isinstance(k, TevaTasks):
            configured_task_factory[k] = v
    
    default_task_factory.update(configured_task_factory)

    if tasks == "all":
        default_task_factory[TevaTasks.WURA]()
        default_task_factory[TevaTasks.EVAL]()
        default_task_factory[TevaTasks.IFT]()
    else:
        selected_sft_tasks = []
        selected_eval_tasks = []
        selected_ift_tasks = []

        all_eval_tasks = TevaTasks.get_evaluation_tasks()
        all_ift_tasks = TevaTasks.get_instruction_tasks()
        all_sft_tasks = TevaTasks.get_supervised_ft_tasks()

        for task in tasks:
            try:
                task_func = default_task_factory[task]
            except KeyError as e:
                raise TaskNotFoundException(task, default_task_factory.keys()) from e
            
            task_name = task_func().name

            mixture_rate_cfg = kwargs.get(f"{task_name}_mixture_cfg", MixtureRateConfig())
            
            if task in all_eval_tasks:
                selected_eval_tasks.append((task_name, get_rate(
                    task_name, **asdict(mixture_rate_cfg)
                )))
            
            elif task in all_ift_tasks:
                selected_ift_tasks.append((task_name, get_rate(
                    task_name, **asdict(mixture_rate_cfg)
                )))

            if task in all_sft_tasks:
                selected_sft_tasks.append((task_name, get_rate(
                    task_name, **asdict(mixture_rate_cfg)
                )))
        
        if len(selected_sft_tasks) > 1:
            seqio.MixtureRegistry.add("teva_sft", selected_sft_tasks, default_rate=1.0)
            logging.info(f"Created `teva_sft` mixture with mixtures/tasks: {selected_sft_tasks}")
        
        if len(selected_eval_tasks) > 1:
            seqio.MixtureRegistry.add("teva_evaluation", selected_eval_tasks, default_rate=1.0)
            logging.info(f"Created `teva_eval` mixture with mixtures/tasks: {selected_sft_tasks}")

        if len(selected_ift_tasks) > 1:
            seqio.MixtureRegistry.add("teva_ift", selected_ift_tasks, default_rate=1.0)
            logging.info(f"Created `teva_ift` mixture with mixtures/tasks: {selected_ift_tasks}")

    logging.info(f"Registed Teva tasks: \n\n{list(seqio.TaskRegistry.names())}\n")
    logging.info(f"Registered Teva mixtures: \n\n{list(seqio.MixtureRegistry.names())}")
