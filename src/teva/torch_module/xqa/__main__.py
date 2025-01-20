import functools
import logging

from teva.torch_module.train import main
from teva.torch_module.arguments import ModelArguments
from teva.torch_module.xqa.arguments import xQADataArguments
from teva.torch_module.xqa.dataset import compute_metrics, get_metrics, preprocess_function
from teva.torch_module.xqa.trainer import xQATrainer

logger = logging.getLogger(__name__)


if __name__ == "__main__":
    main(
        preprocess_function=preprocess_function,
        compute_metrics_function=functools.partial(compute_metrics, metric=get_metrics()),
        data_arguments=xQADataArguments,
        model_arguments=ModelArguments,
        trainer_cls=xQATrainer
    )
