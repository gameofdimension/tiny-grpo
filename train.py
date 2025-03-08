def optimize_model_memory(model):
    """
    Optimizes the model to use less memory during training.

    Args:
        model: The language model to optimize.

    Returns:
        The optimized model.

    Explanation:
        1. Sets the model to training mode.
        2. Disables KV caching to save memory.
        3. Enables gradient checkpointing to trade computation for memory.
        4. Ensures that input embeddings require gradients:
           - Either uses the built-in method if available.
           - Or adds a forward hook to the input embeddings layer.
        5. Returns the optimized model ready for memory-efficient training.
    """
    model.train()
    model.config.use_cache = False

    # First ensure inputs will require gradients
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    else:
        def make_inputs_require_grad(module, input, output):
            output.requires_grad_(True)
        model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    # Then enable gradient checkpointing
    model.gradient_checkpointing_enable()

    return model

# Main execution
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using primary device: {device}")

model_name = "Qwen/Qwen2.5-1.5B-Instruct"
output_dir = "math_solver_model"

print("Downloading model...")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
print("Model downloaded")

tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.eos_token_id
model.config.eos_token_id = tokenizer.eos_token_id

num_gpus = torch.cuda.device_count()
print(f"Detected {num_gpus} GPUs")
device_ids = list(range(num_gpus)) if num_gpus > 1 else None

all_data = prepare_dataset("train")
random.shuffle(all_data)
size_of_eval_data = 30 # change to a smaller value to save time or to a larger number for a more reliable estimate
eval_data = all_data[:size_of_eval_data]
train_data = all_data[size_of_eval_data:]

print("\nInitial model evaluation before finetuning:")
pre_grpo_accuracy = evaluate_model(model, tokenizer, eval_data, device)
print(f"Pre-GRPO Accuracy: {pre_grpo_accuracy:.2f}%")

model = optimize_model_memory(model)

print("\nStarting RL fine-tuning using GRPO...")
# This config was tested on a 8xA100 node, where each A100 is has 80GB of VRAM
training_config = {
    'num_iterations': 1,
    'num_steps': 500,
    'batch_size': 7, # reduce if you have fewer GPUs
    'num_generations': 12, # reduce if you have GPUs with less VRAM
    'max_completion_length': 400, # reduce if you have GPUs with less VRAM
    'beta': 0.04,
    'learning_rate': 5e-6,
    'mu': 1,
    'epsilon': 0.1
}

# Initialize Weights & Biases
wandb.init(project=os.environ["WANDB_PROJECT"], reinit=True)
print("Weights & Biases initialized.")

model = train_with_grpo(
    model=model,
    tokenizer=tokenizer,
    train_data=train_data,
    reward_function=combined_reward,
    device_ids=device_ids,
    **training_config
)

wandb.finish()
print("Training completed and wandb run finished.")

print("\nFinal model evaluation after GRPO RL fine-tuning:")
post_grpo_accuracy = evaluate_model(model, tokenizer, eval_data, device)
print(f"Post-GRPO Accuracy: {post_grpo_accuracy:.2f}%")

print("\nSaving GRPO fine-tuned model...")
model.save_pretrained("grpo_finetuned_model")
tokenizer.save_pretrained("grpo_finetuned_model")