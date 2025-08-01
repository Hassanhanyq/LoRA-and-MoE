import torch
import torch.nn as nn
import torch.nn.functional as F
from .lora_expert import LoRAExpert

class MoEUpProjWithLoRA(nn.Module):
    def __init__(self, original_up_proj, num_experts=4, hidden_size=4096, adapter_rank=8, alpha=8):
        super().__init__()
        self.num_experts = num_experts
        self.hidden_size = hidden_size

        self.original_up_proj = original_up_proj
        for param in self.original_up_proj.parameters():
            param.requires_grad = False  #freeze base

        self.experts = nn.ModuleList([
            LoRAExpert(hidden_size, adapter_rank, alpha) for _ in range(num_experts)
        ])
        self.gate = nn.Linear(hidden_size, num_experts, bias=False)
        self.expert_bias = nn.Parameter(torch.zeros(num_experts))

        #tracking
        self.routing_stats = []          
        self.expert_token_counts = []    

    def forward(self, x):
        """
        x: [B, T, H]
        """
        B, T, H = x.shape
        x_flat = x.view(-1, H)  

        
        gate_logits = self.gate(x_flat)+self.expert_bias              
        probs = F.softmax(gate_logits, dim=-1)          

        
        top1_idx = torch.argmax(probs, dim=-1)          

        
        hard_one_hot = F.one_hot(top1_idx, num_classes=self.num_experts).float()  
        gate_mix = (hard_one_hot - probs).detach() + probs  

        
        self.routing_stats.append(top1_idx.detach().cpu())
        counts = torch.bincount(top1_idx, minlength=self.num_experts)
        self.expert_token_counts.append(counts.detach().cpu())

        
        base_out = self.original_up_proj(x)  

        
        sorted_idx = torch.argsort(top1_idx)
        x_sorted = x_flat[sorted_idx]         
        top1_sorted = top1_idx[sorted_idx]     

        expert_outputs = torch.zeros_like(x_flat)
        #scatter/gather probs works better here this should be fixed
        cursor = 0
        for expert_id in range(self.num_experts):
            mask = (top1_sorted == expert_id)
            cnt = mask.sum().item()
            if cnt == 0:
                continue
            expert_input = x_sorted[cursor:cursor + cnt]  
            expert_output = self.experts[expert_id](expert_input)  
            expert_outputs[cursor:cursor + cnt] = expert_output
            cursor += cnt

        
        restore_idx = torch.argsort(sorted_idx)
        updated = expert_outputs[restore_idx].view(B, T, H)  

        return base_out + updated

    def update_expert_bias(self, update_rate=0.001):
        """
        Calculates the load violation error and updates the expert bias.
        This method will be called from the training loop.
        """
        
        if not self.expert_token_counts:
            return

        
        token_counts = self.expert_token_counts[-1].float()
        total_tokens = token_counts.sum()
        
        if total_tokens > 0:
            
            avg_tokens = total_tokens / self.num_experts

            
            load_error = token_counts - avg_tokens

            
            with torch.no_grad():
                self.expert_bias.data.add_(update_rate * torch.sign(load_error))