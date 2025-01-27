from dataclasses import dataclass, field
from typing import Optional

from teva.torch_module.arguments import DataArguments


@dataclass
class MCQADataArguments(DataArguments):
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    context_column: Optional[str] = field(
        default="context",
        metadata={"help": "The name of the column in the datasets containing the contexts (for question answering)."},
    )
    question_column: Optional[str] = field(
        default="question",
        metadata={"help": "The name of the column in the datasets containing the questions (for question answering)."},
    )
    answer_column: Optional[str] = field(
        default="answers",
        metadata={"help": "The name of the column in the datasets containing the answers (for question answering)."},
    )
    option_columns: Optional[list[str]] = field(
        default=None,
        metadata={"help": "Ordered list of option columns"}
    )
