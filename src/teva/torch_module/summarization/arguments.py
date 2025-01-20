from dataclasses import dataclass, field
from typing import Optional

from transformers import Seq2SeqTrainingArguments

from teva.torch_module.arguments import DataArguments


@dataclass
class SummarizationTrainingArguments(Seq2SeqTrainingArguments):
    sampling_factor: Optional[float] = field(default=0.3, metadata={"help": "Upsampling factor for lower resource dataset configurations."})

    # Additional generation args
    length_penalty: Optional[float] = field(default=0.6, metadata={"help": "Length penalty for generation"})


@dataclass
class SummarizationDataArguments(DataArguments):
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    lang: Optional[str] = field(default=None, metadata={"help": "Language id for summarization."})
    text_column: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the column in the datasets containing the full texts (for summarization)."},
    )
    summary_column: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the column in the datasets containing the summaries (for summarization)."},
    )
    max_source_length: Optional[int] = field(
        default=1024,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    max_target_length: Optional[int] = field(
        default=128,
        metadata={
            "help": (
                "The maximum total sequence length for target text after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    val_max_target_length: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "The maximum total sequence length for validation target text after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded. Will default to `max_target_length`. "
                "This argument is also used to override the ``max_length`` param of ``model.generate``, which is used "
                "during ``evaluate`` and ``predict``."
            )
        },
    )
    source_prefix: Optional[str] = field(
        default=None, metadata={"help": "A prefix to add before every source text (useful for T5 models)."}
    )
    forced_bos_token: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The token to force as the first generated token after the decoder_start_token_id. "
                "Useful for multilingual models like mBART where the first generated token"
                "needs to be the target language token (Usually it is the target language token)"
            )
        },
    )
