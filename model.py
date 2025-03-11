from collections import namedtuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from parallelize import apply_ac, apply_compile, apply_fsdp
from utils import is_main_process
from vlogging import logger


def load_tokenizer(ckpt_path: str):
    tokenizer = AutoTokenizer.from_pretrained(ckpt_path, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def prepare_actor_rollout_model(
    ckpt_path: str,
    eos_token_id,
    compile: bool,
    sac,
    mesh,
):
    dtype = torch.float32
    transformer = AutoModelForCausalLM.from_pretrained(
        ckpt_path,
        torch_dtype=torch.bfloat16
    )
    transformer.config.pad_token_id = eos_token_id
    transformer.config.eos_token_id = eos_token_id

    transformer = transformer.to(dtype=dtype)
    transformer.requires_grad_(True)
    transformer.train()

    total_params = sum([p.numel() for p in transformer.parameters()])
    trainable_params = sum(
        [p.numel() for p in transformer.parameters() if p.requires_grad])

    if is_main_process():
        first_param = next(transformer.parameters())
        logger.info(f"init device: {first_param.device}")
        logger.info(f"dtype: {first_param.dtype}")
        logger.info(f"total params: {total_params}")
        logger.info(f"trainable params: {trainable_params}")

    if sac == 'op':
        ac_config = namedtuple("AcConfig", ["mode", "selective_ac_option"])(
            **{"mode": "selective", "selective_ac_option": "op"}
        )
        apply_ac(transformer, ac_config=ac_config)

    if compile:
        if sac != 'no' and sac != 'op':
            torch._functorch.config.activation_memory_budget = float(sac)
            logger.info(
                f"will apply automatic sac through torch.compile with budget: {sac}")
        apply_compile(transformer)

    apply_fsdp(transformer, mesh)

    if is_main_process():
        logger.info(f"model: {transformer}")

    torch.cuda.empty_cache()
    return transformer


def prepare_ref_model(
    ckpt_path: str,
    eos_token_id,
    compile: bool,
    mesh,
):
    dtype = torch.float32
    transformer = AutoModelForCausalLM.from_pretrained(
        ckpt_path,
        torch_dtype=torch.bfloat16
    )
    transformer.config.pad_token_id = eos_token_id
    transformer.config.eos_token_id = eos_token_id

    transformer = transformer.to(dtype=dtype)
    transformer.requires_grad_(False)
    transformer.eval()

    total_params = sum([p.numel() for p in transformer.parameters()])
    if is_main_process():
        first_param = next(transformer.parameters())
        logger.info(f"init device: {first_param.device}")
        logger.info(f"dtype: {first_param.dtype}")
        logger.info(f"total params: {total_params}")

    if compile:
        apply_compile(transformer)

    apply_fsdp(transformer, mesh)

    if is_main_process():
        logger.info(f"model: {transformer}")

    torch.cuda.empty_cache()
    return transformer
