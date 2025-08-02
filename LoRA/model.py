from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from MoELoRA import MoEUpProjWithLoRA
model_name = "Qwen/Qwen3-0.6B"



tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto",
    trust_remote_code=True
)
target_layers = [14, 15, 16, 17, 18]
adapter_rank = 8
num_experts = 4
alpha = 8
print(model)
for i in target_layers:
    block = model.model.layers[i]

    
    original_up_proj = block.mlp.up_proj

    
    hidden_size = original_up_proj.in_features

    
    block.mlp.up_proj = MoEUpProjWithLoRA(
        original_up_proj=original_up_proj,
        num_experts=num_experts,
        hidden_size=hidden_size,
        adapter_rank=adapter_rank,
        alpha=alpha
    )

    print(f"Injected MoE LoRA into layer {i} (up_proj)")

print(" All target layers patched.")
