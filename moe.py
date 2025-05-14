"""
A lot of this code is from Mergoo's implementation, except this is sequence level not token level
"""
from torch import nn
import torch

# This will convert nn.Linear to our MoE class (in merge_moe.py)
def convert_linear_to_moe(name, config, layer_ind, in_features, out_features, bias=True):
    if layer_ind in config.router_layers_index and name in config.router_layers:
        return MoE(in_features, out_features)

# Following Mixtral's approach to building a router
class MoE(nn.Module):
    def __init__(self, hidden_size, num_experts=8):
        super().__init__()
        self.gating_vector = nn.Linear(hidden_size, num_experts)
        self.experts = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(num_experts)])

    def forward(self, hidden_states):
        # hidden states: [batch_size, seq_len, hidden_size]
        sequences = hidden_states.mean(dim=1) # Retrieve sequences [batch_size, hidden_size]
        scores = self.gating_vector(sequences) # Pass through gating vector
        weights, selected_experts = torch.topk(scores, 2) # Select top 2 experts

        results = torch.zeros(
            (sequences.shape[0], sequences.shape[1])
        )

        # For each expert, add the weighted output to the results
        for ind, expert in enumerate(self.experts):
            batch_ind, seq_ind = torch.where(selected_experts == ind)
            if batch_ind.numel() != 0:
                results[batch_ind] += expert(sequences[batch_ind]) * weights[batch_ind, seq_ind].unsqueeze(-1)
        return results






