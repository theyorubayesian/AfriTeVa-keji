import functools
import logging

from teva.torch_module.train import main
from teva.torch_module.arguments import ModelArguments
from teva.torch_module.summarization.arguments import SummarizationDataArguments, SummarizationTrainingArguments
from teva.torch_module.summarization.dataset import compute_metrics, dataset_provider, get_metrics, preprocess_function
from teva.torch_module.summarization.trainer import S2STrainer

logger = logging.getLogger(__name__)


if __name__ == "__main__":
    main(
        dataset_provider=dataset_provider,
        preprocess_function=preprocess_function,
        compute_metrics_function=functools.partial(compute_metrics, metric=get_metrics()),
        training_arguments=SummarizationTrainingArguments,
        data_arguments=SummarizationDataArguments,
        model_arguments=ModelArguments,
        trainer_cls=S2STrainer
    )
