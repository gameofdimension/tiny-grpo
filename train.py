import os
import sys

import torch
import torch.distributed as dist
import wandb
from torch.distributed.device_mesh import init_device_mesh

from data import make_dataloaders
from eval import evaluate_model
from loss import train_with_grpo
from model import (load_tokenizer, prepare_actor_rollout_model,
                   prepare_ref_model)
from reward import combined_reward
from utils import init_distributed, is_main_process, set_random_seed
from vlogging import init_logger, logger


def main():
    world_size, rank, local_rank = init_distributed()
    model_name = sys.argv[1]

    tokenizer = load_tokenizer(model_name)
    mesh = init_device_mesh(device_type='cuda', mesh_shape=(dist.get_world_size(),))
    model = prepare_actor_rollout_model(
        ckpt_path=model_name,
        eos_token_id=tokenizer.eos_token_id,
        compile=False,
        sac='no',
        mesh=mesh,
    )
    ref_model = prepare_ref_model(
        ckpt_path=model_name,
        eos_token_id=tokenizer.eos_token_id,
        compile=False,
        mesh=mesh,
    )

    batch_size = 1
    train_data_loader, eval_data_loader = make_dataloaders(
        data_path=sys.argv[2],
        distributed=True,
        dp_degree=world_size,
        dp_rank=rank,
        train_batch_size=batch_size,
        num_workers=4
    )

    current_device = torch.device(f"cuda:{torch.cuda.current_device()}")
    do_eval = sys.argv[3].lower() == "yes"
    if do_eval:
        logger.info("Initial model evaluation before finetuning:")
        pre_grpo_accuracy = evaluate_model(model, tokenizer, eval_data_loader, current_device)
        logger.info(f"Pre-GRPO Accuracy: {pre_grpo_accuracy:.2f}%")
    logger.info("Starting RL fine-tuning using GRPO...")
    # This config was tested on a 8xA100 node, where each A100 is has 80GB of VRAM
    training_config = {
        'num_iterations': 1,
        'num_steps': 500,
        'num_generations': 12,  # reduce if you have GPUs with less VRAM
        'max_completion_length': 400,  # reduce if you have GPUs with less VRAM
        'beta': 0.04,
        'learning_rate': 1e-6,
        'mu': 1,
        'epsilon': 0.1
    }

    # Initialize Weights & Biases
    if is_main_process():
        wandb.init(project=os.environ["WANDB_PROJECT"], reinit=True)
    logger.info("Weights & Biases initialized.")

    model = train_with_grpo(
        model=model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        train_data=train_data_loader,
        reward_function=combined_reward,
        **training_config
    )

    if is_main_process():
        wandb.finish()
    logger.info("Training completed and wandb run finished.")

    if do_eval:
        logger.info("Final model evaluation after GRPO RL fine-tuning:")
        post_grpo_accuracy = evaluate_model(model, tokenizer, eval_data_loader, current_device)
        logger.info(f"Post-GRPO Accuracy: {post_grpo_accuracy:.2f}%")


if __name__ == "__main__":
    # Call the function to set random seed for reproducibility
    set_random_seed(42)
    init_logger()
    main()
