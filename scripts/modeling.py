from unsloth import FastLanguageModel, FastModel
from peft import LoraConfig
from transformers import AutoTokenizer


def build_model_and_tokenizer(base_repo, max_seq_len, lora_cfg, attn_impl="flash_attention_2", use_unsloth_compile=True):
    # 4-bit quantized base for QLoRA

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_repo,
        max_seq_length=max_seq_len,
        load_in_4bit=True,
        attn_implementation=attn_impl,
    )

    # Ensure tokenizer has PAD
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # peft_cfg = LoraConfig(
    #     r=lora_cfg["r"],
    #     lora_alpha=lora_cfg["alpha"],
    #     lora_dropout=lora_cfg["dropout"],
    #     bias=lora_cfg["bias"],
    #     target_modules=lora_cfg["target_modules"],
    #     task_type="CAUSAL_LM",
    # )

    # Apply LoRA (Unsloth helper)
    model = FastModel.get_peft_model(
        model,
        finetune_vision_layers=False,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        **{k: lora_cfg[k] for k in ["r"]} # r used internally; alpha/dropout handled by PEFT
    )

    #model.add_adapter(peft_cfg)

    # Unsloth speedups (compile) for training; disable at inference/export
    if use_unsloth_compile:
        FastLanguageModel.for_training(model)   # enables fwd/bwd speedups

    return model, tokenizer