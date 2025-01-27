from teva.torch_module.train import main
from teva.torch_module.arguments import ModelArguments
from teva.torch_module.classification.dataset import compute_classification_metrics
from teva.torch_module.mcqa.arguments import MCQADataArguments
from teva.torch_module.mcqa.dataset import preprocess_function


if __name__ == "__main__":
    main(
        data_arguments=MCQADataArguments,
        model_arguments=ModelArguments,
        preprocess_function=preprocess_function,
        compute_metrics_function=compute_classification_metrics
    )
