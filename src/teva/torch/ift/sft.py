from transformers import AutoTokenizer, T5ForConditionalGeneration
from trl import (
    ModelConfig,
    SFTConfig,
    SFTTrainer,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config
)

from .arguments import ScriptArguments
from .aya_mixture import SeqioDataset
from teva.teva_tasks import TevaTasks


def main():
    parser = TrlParser((ScriptArguments, SFTConfig, ModelConfig))
    script_args, training_args, model_config = parser.parse_args_and_config()

    tokenizer = AutoTokenizer.from_pretrained(
        model_config.model_name_or_path, trust_remote_code=model_config.trust_remote_code, use_fast=True
    )

    # Set pad token?
    # tokenizer.pad_token = tokenizer.eos_token
    quantization_config = get_quantization_config(model_config)

    model = T5ForConditionalGeneration.from_pretrained(
        model_config.model_name_or_path,
        revision=model_config.model_revision,
        trust_remote_code=model_config.trust_remote_code,
        attn_implementation=model_config.attn_implementation,
        use_cache=False if training_args.gradient_checkpointing else True,
        torch_dtype=model_config.torch_dtype,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config
    )

    dataset = SeqioDataset(task=script_args.teva_tasks, copy_pretokenized=True)

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset.get_dataset("train"),
        eval_dataset=dataset.get_dataset("validation") if training_args.eval_strategy != "no" else None,
        processing_class=tokenizer,
        peft_config=get_peft_config(model_config)
    )

    trainer.train()
    trainer.save_model(training_args.output_dir)
