from string import Template

from datasets.formatting.formatting import LazyBatch
from transformers import BatchEncoding, PreTrainedTokenizerBase

from teva.torch_module.mcqa.arguments import MCQADataArguments


MCQA_TEMPLATE = Template(
    """
    Passage:
    
    ${passage}
    
    Question: ${question}
    
    Options:
    ${options_text}
    """
)

MCQA_PASSAGE_TEMPLATE = Template("Passage: ${passage}")
MCQA_QUESTION_AND_OPTION_TEMPLATE = Template(
    """
    Question: ${question}
    
    Options:
    ${options_text}
    """
)


def maybe_format_labels(labels: list[str | int]) -> list[str]:
    if isinstance(labels[0], int) or labels[0].isdigit():
        return [chr(64 + int(label)) for label in labels]
    
    assert set(labels).issubset(["A", "B", "C", "D"])
    return labels


def preprocess_function(
    examples: LazyBatch,
    data_args: MCQADataArguments,
    tokenizer: PreTrainedTokenizerBase,
    **kwargs
) -> BatchEncoding:
    contexts = [
        MCQA_PASSAGE_TEMPLATE.substitute(passage=p)
        for p in examples[data_args.context_column]
    ]
    questions = examples[data_args.question_column]

    qna = []
    for i in range(len(contexts)):
        options_text = "\n".join(
            f"{chr(65 + idx)}. {examples[col][i]}"
            for idx, col in enumerate(data_args.option_columns)
        )

        formatted_qna = MCQA_QUESTION_AND_OPTION_TEMPLATE.substitute(
            passage=contexts[i],
            question=questions[i],
            options_text=options_text,
        )
        qna.append(formatted_qna)

    model_inputs = tokenizer(
        contexts,
        qna,
        max_length=data_args.max_seq_length,
        padding=data_args.padding,
        truncation=False
    )

    labels = tokenizer(maybe_format_labels(examples[data_args.answer_column]), max_length=2)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs
