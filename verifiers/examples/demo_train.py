from trl import GRPOConfig

import verifiers as vf
from verifiers.tools import python
from verifiers.utils import preprocess_dataset

model_name = "Qwen/Qwen2.5-1.5B-Instruct"

"""
2-GPU training (single node, 1 training + 1 inference)

CUDA_VISIBLE_DEVICES=0 uv run -m verifiers.inference.vllm_serve --model 'Qwen/Qwen2.5-1.5B-Instruct' --max_model_len 4096 --dtype bfloat16 --gpu_memory_utilization 0.95 --enable_prefix_caching True
CUDA_VISIBLE_DEVICES=1 accelerate launch --num-processes 1 --config-file configs/zero3.yaml verifiers/examples/demo_train.py
---
4-GPU training (single node, 2 training + 2 inference)

CUDA_VISIBLE_DEVICES=0,1 python verifiers/inference/vllm_serve.py --model 'Qwen/Qwen2.5-1.5B-Instruct' --max_model_len 4096 --dtype bfloat16 --gpu_memory_utilization 0.95 --enable_prefix_caching True
CUDA_VISIBLE_DEVICES=2,3 accelerate launch --num-processes 2 --config-file configs/zero3.yaml verifiers/examples/demo_train.py
"""

SIMPLE_PROMPT = """
You are a helpful assistant. In each turn, think step-by-step inside <think>...</think> tags, then give your final answer inside <answer>...</answer> tags.
"""
import torch
from torch.utils.checkpoint import checkpoint
from functools import partial

# This function will be model-specific. You need to find the correct attribute path
# to the attention layers for Qwen2.5-1.5B-Instruct.
# Common paths are model.transformer.h or model.model.layers
def patch_attention_for_checkpointing(model_to_patch, model_config):
    """Patches attention layers in the model for selective gradient checkpointing."""

    # Example for a generic Hugging Face transformer architecture.
    # You WILL need to adjust `model_to_patch.model.layers` and `layer.self_attn`
    # based on Qwen2.5-1.5B-Instruct's actual structure.
    if not hasattr(model_to_patch, 'model') or not hasattr(model_to_patch.model, 'layers'):
        print("Could not find model.layers. Check model architecture for Qwen2.5-1.5B-Instruct.")
        return

    print(f"Attempting to patch attention layers for gradient checkpointing...")
    layers_patched = 0
    for i, layer in enumerate(model_to_patch.model.layers):
        if hasattr(layer, 'self_attn'):
            # The actual attention computation might be within a sub-module of self_attn
            # For example, it could be layer.self_attn.o_proj or similar that you want to checkpoint *around*,
            # or the core attention mechanism itself.
            # Let's assume layer.self_attn.forward() is the boundary for checkpointing.

            original_forward = layer.self_attn.forward

            # `use_reentrant=False` is generally recommended for better performance if supported
            # by your PyTorch version and model for checkpointing.
            # The default for torch.utils.checkpoint.checkpoint is use_reentrant=True.
            # Hugging Face Transformers often default to use_reentrant=False when they internally
            # enable gradient checkpointing.
            # `model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})`
            # is how HF Trainer does it.

            # We need to make sure the arguments are passed correctly.
            # The `checkpoint` function expects the function first, then its args.
            def checkpointed_forward(*args, **kwargs):
                # print(f"Calling checkpointed attention in layer {i}")
                return checkpoint(original_forward, *args, **kwargs, use_reentrant=False)

            layer.self_attn.forward = checkpointed_forward
            layers_patched += 1
        else:
            print(f"Layer {i} does not have self_attn attribute.")
    if layers_patched > 0:
        print(f"Successfully patched {layers_patched} attention layers for gradient checkpointing.")
    else:
        print("No attention layers were patched. Please verify module names and model structure.")

# Before creating the trainer, patch the model if you want to use custom checkpointing
# Make sure training_args.gradient_checkpointing is True if you want HF to handle the rest,
# or False if you are *only* relying on your manual patch.
# For this specific selective attention patching, we'll assume HF's main GC is also active.

dataset = preprocess_dataset("math", "train", n=1000)

#vf_env = vf.SingleTurnEnv(
vf_env = vf.DoubleCheckEnv(
    dataset=dataset,
    system_prompt=SIMPLE_PROMPT,
    few_shot=[]
)
print(vf_env.system_prompt)

model, tokenizer = vf.get_model_and_tokenizer(model_name)
patch_attention_for_checkpointing(model, model.config)
run_name = "demo-grpo_" + model_name.split("/")[-1].lower()

training_args=GRPOConfig(
    output_dir=f"outputs/{run_name}",
    run_name=run_name,
    learning_rate=1e-6,
    lr_scheduler_type="constant",
    num_train_epochs=1,
    temperature=1.0,
    max_steps=100,
    bf16=True,
    max_grad_norm=0.1,
    num_iterations=1,
    beta=0,
    max_prompt_length=512,
    max_completion_length=1024,
    per_device_train_batch_size=2,
    num_generations=4,
    gradient_accumulation_steps=1,
    gradient_checkpointing=True,
    save_strategy="steps",
    save_steps=100,
    save_only_model=True,
    use_vllm=True,
    # vllm_kwargs={"gpu_memory_utilization": 0.4, "swap_space": 4},
     vllm_gpu_memory_utilization=0.4,
    logging_steps=1,
    log_on_each_node=False,
    log_completions=True,
    report_to="wandb",
    reward_weights=vf_env.get_reward_weights()
)

trainer = vf.GRPOEnvTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=vf_env.get_reward_funcs(),
    env=vf_env,
    args=training_args,
    train_dataset=vf_env.get_dataset()
)
trainer.train()