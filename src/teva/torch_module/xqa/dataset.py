from typing import Literal

from datasets import Dataset
from evaluate import load as load_metric, EvaluationModule
from transformers import EvalPrediction, PreTrainedTokenizer, BatchEncoding
from transformers.trainer_utils import EvalLoopOutput

from teva.torch_module.xqa.arguments import DataTrainingArguments


def preprocess_function(
    dataset: Dataset,
    tokenizer: PreTrainedTokenizer, 
    data_args: DataTrainingArguments,
    mode: Literal["train", "validation", "prediction"] = "train",
) -> BatchEncoding:
    model_inputs = tokenizer(
        dataset["question"],
        dataset["context"],
        max_length=data_args.max_seq_length,
        padding=data_args.padding,
        stride=data_args.doc_stride,
        truncation=True,
        return_overflowing_tokens=True,
        return_offsets_mapping=True
    )

    offset_mapping = model_inputs.pop("offset_mapping")
    sample_map = model_inputs["overflow_to_sample_mapping"]
    answers = dataset["answers"]

    labels = []

    for i, offset in enumerate(offset_mapping):
        sample_idx = sample_map[i]
        answer = answers[sample_idx]

        try:
            start_char = answer["answer_start"][0]
            end_char = answer["answer_start"][0] + len(answer["text"][0])
        except IndexError:
            labels.append(data_args.no_answer_string)
            continue

        sequence_ids = model_inputs.sequence_ids(i)

        # Find start and end of context
        idx = 0
        while sequence_ids[idx] != 1:
            idx += 1
        context_start = idx
        while sequence_ids[idx] == 1:
            idx += 1
        context_end = idx - 1

        if offset[context_start][0] > start_char or offset[context_end][1] < end_char:
            # answer is not fully inside the context
            labels.append(data_args.no_answer_string)
        else:
            labels.append(answers[sample_map[i]]["text"][0])

    labels = tokenizer(
        labels,
        max_length=data_args.max_target_length,
        padding=data_args.padding,
        truncation=True
    )

    # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
    # padding in the loss.
    if data_args.padding == "max_length" and data_args.ignore_pad_token_for_loss:
        labels = [
            [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
        ]
    else:
        labels = [row["input_ids"] for row in labels]

    model_inputs["labels"] = labels
    return model_inputs


def post_processing_function(
    examples: Dataset, features: Dataset, outputs: EvalLoopOutput,  data_args: DataTrainingArguments,
    tokenizer: PreTrainedTokenizer, stage="eval",
):
    # Decode the predicted tokens.
    preds = outputs.predictions
    if isinstance(preds, tuple):
        preds = preds[0]
    
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    # Build a map example to its corresponding features.
    example_id_to_index = {k: i for i, k in enumerate(examples["id"])}
    feature_per_example = {example_id_to_index[feature["example_id"]]: i for i, feature in enumerate(features)}
    predictions = {}
    # Let's loop over all the examples!
    for example_index, example in enumerate(examples):
        # This is the index of the feature associated to the current example.
        try:
            feature_index = feature_per_example[example_index]
            predictions[example["id"]] = decoded_preds[feature_index]
        except KeyError:
            continue

    # Format the result to the format the metric expects.
    if data_args.version_2_with_negative:
        formatted_predictions = [
            {"id": k, "prediction_text": v, "no_answer_probability": 0.0} for k, v in predictions.items()
        ]
    else:
        formatted_predictions = [{"id": k, "prediction_text": v} for k, v in predictions.items()]

    references = [{"id": ex["id"], "answers": ex[data_args.answer_column]} for ex in examples if ex["context"] is not None]
    return EvalPrediction(predictions=formatted_predictions, label_ids=references)


def get_metrics():
    return load_metric("squad_v2")


def compute_metrics(p: EvalPrediction, metric: EvaluationModule):
    return metric.compute(predictions=p.predictions, references=p.label_ids)
