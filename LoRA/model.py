import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, TrainerCallback
from trl import SFTTrainer
from datasets import Dataset
import json
import logging
from MoELoRA import MoEUpProjWithLoRA


logger = logging.getLogger(__name__)


model_name = "Qwen/Qwen3-0.6B"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)


target_layers = [14, 15, 16, 17, 18]
adapter_rank = 8
num_experts = 4
alpha = 8

for i in target_layers:
    block = model.model.layers[i]
    original_up_proj = block.mlp.up_proj
    hidden_size = original_up_proj.in_features
    block.mlp.up_proj = MoEUpProjWithLoRA(
        original_up_proj=original_up_proj,
        num_experts=num_experts,
        adapter_rank=adapter_rank,
        alpha=alpha,
        dtype=model.dtype
    )
    print(f"Injected MoE LoRA into layer {i} (up_proj)")

print("All target layers patched.")




class MoEBalanceAndTrackingCallback(TrainerCallback):
    def _find_moe_layers(self, model):
        moe_layers = {}
        for name, module in model.named_modules():
            if isinstance(module, MoEUpProjWithLoRA):
                moe_layers[name] = module
        return moe_layers

    def on_step_end(self, args, state, control, **kwargs):
        model = kwargs['model']
        
        if hasattr(model, 'base_model'):
            model = model.base_model
        if hasattr(model, 'model'):
            model = model.model
            
        moe_layers = self._find_moe_layers(model)
        
        if not moe_layers:
            return

        for _, moe_layer in moe_layers.items():
            moe_layer.update_expert_bias(update_rate=0.001)
            
        if state.global_step % args.logging_steps != 0:
            return

        logs = {}
        for name, moe_layer in moe_layers.items():
            if moe_layer.expert_token_counts:
                counts = torch.stack(moe_layer.expert_token_counts).sum(dim=0).float()
                total_tokens = counts.sum().item()
                
                if 'up_proj' in name:
                    parts = name.split('.')
                    layer_idx = None
                    for part in parts:
                        if part.isdigit():
                            layer_idx = part
                            break
                    
                    if layer_idx:
                        for expert_id, count in enumerate(counts):
                            logs[f"moe/layer_{layer_idx}/expert_{expert_id}_tokens"] = count.item()
                        logs[f"moe/layer_{layer_idx}/total_tokens"] = total_tokens
                        logs[f"moe/layer_{layer_idx}/avg_tokens"] = total_tokens / moe_layer.num_experts

                
                moe_layer.expert_token_counts.clear()
                moe_layer.routing_stats.clear()
        
        
        if logs:
            current_logs = kwargs.get('logs', {})
            current_logs.update(logs)



file_path = "allqa.json"
with open(file_path, 'r', encoding='utf-8') as f:
    raw_data = json.load(f)



print("Starting SFT...")
sft_data = []
for item in raw_data:
    prompt = item["prompt"]
    chosen = item["chosen"]
    formatted_text = f"### Human:\n{prompt}\n### Assistant:\n{chosen}"
    sft_data.append({"text": formatted_text})
train_dataset = Dataset.from_list(sft_data)
training_args = TrainingArguments(
    output_dir="./results_sft",
    num_train_epochs=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    optim="paged_adamw_32bit",
    save_strategy="epoch",
    logging_steps=10,
    learning_rate=2e-4,
    bf16=True,
    max_steps=-1,
    gradient_checkpointing=True,
)
sft_trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    args=training_args,
    callbacks=[MoEBalanceAndTrackingCallback()]
)
sft_trainer.train()
sft_trainer.save_model("./sft_model")
print("SFT model saved to ./sft_model")