import functools
import logging

from teva.torch_module.train import main
from teva.torch_module.translation.arguments import DataTrainingArguments, ModelArguments
from teva.torch_module.translation.dataset import compute_metrics, get_metrics, preprocess_function

logger = logging.getLogger(__name__)


if __name__ == "__main__":
    main(
        data_arguments=DataTrainingArguments,
        model_arguments=ModelArguments,
        preprocess_function=preprocess_function,
        compute_metrics_function=functools.partial(compute_metrics, metrics=get_metrics())
    )
