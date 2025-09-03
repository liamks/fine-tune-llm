from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import os
import torch

import os, torch
if torch.cuda.is_available():
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)   # <-- kills the NCCL “unknown device” warning




def build_model_and_tokenizer(base_repo, max_seq_len, lora_cfg, attn_impl="flash_attention_2", use_unsloth_compile=True, use_ddp=True,):
    # 4-bit quantized base for QLoRA

    tokenizer = AutoTokenizer.from_pretrained(base_repo, use_fast=True)
        # Ensure tokenizer has PAD
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    

    # model, tokenizer = FastLanguageModel.from_pretrained(
    #     model_name=base_repo,
    #     max_seq_length=max_seq_len,
    #     load_in_4bit=True,
    #     attn_implementation=attn_impl,
    #      device_map={ "": local_rank }
    # )

    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    # --- Device handling ---
    # With DDP (accelerate launch), DO NOT use device_map for sharding.
    # Let Accelerate/Trainer place the whole model on the local rank.

    device_map = None if use_ddp else {"": 0}  # all on one GPU if not using DDP

    model = AutoModelForCausalLM.from_pretrained(
        base_repo,
        # quantization_config=bnb_cfg,
        dtype=torch.bfloat16,
        attn_implementation=attn_impl,
        device_map=device_map
    )

    # Training-time settings
    model.config.use_cache = False
    # Gradient checkpointing (saves VRAM; works fine with 4-bit)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})


    # --- Prepare for k-bit training ---
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )

    peft_cfg = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["alpha"],
        lora_dropout=lora_cfg["dropout"],
        bias=lora_cfg["bias"],
        target_modules=lora_cfg["target_modules"],
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, peft_cfg)

    # Apply LoRA (Unsloth helper)
    # model = FastModel.get_peft_model(
    #     model,
    #     finetune_vision_layers=False,
    #     finetune_language_layers=True,
    #     finetune_attention_modules=True,
    #     finetune_mlp_modules=True,
    #     **{k: lora_cfg[k] for k in ["r"]} # r used internally; alpha/dropout handled by PEFT
    # )

    #model.add_adapter(peft_cfg)

    # Unsloth speedups (compile) for training; disable at inference/export
    # if use_unsloth_compile:
    #     FastLanguageModel.for_training(model)   # enables fwd/bwd speedups

    return model, tokenizer