from teva.torch_module.train import main
from teva.torch_module.arguments import ModelArguments
from teva.torch_module.classification.arguments import ClassificationDataArguments
from teva.torch_module.classification.dataset import compute_classification_metrics, preprocess_function


if __name__ == "__main__":
    main(
        data_arguments=ClassificationDataArguments,
        model_arguments=ModelArguments,
        preprocess_function=preprocess_function,
        compute_metrics_function=compute_classification_metrics
    )
