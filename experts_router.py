from transformers import AutoTokenizer, AutoModelForCausalLM
from adapters import DoubleSeqBnConfig, AutoAdapterModel
from torch import nn
import torch

model_name = "meta-llama/Llama-3.2-3B-Instruct"
model = AutoAdapterModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name, padding=True)
tokenizer.pad_token = tokenizer.eos_token

# Houlby Adapter Config according to https://arxiv.org/pdf/1902.00751
config = DoubleSeqBnConfig()

# Create 8 experts
for i in range(8):
    expert_name = f"expert_{i}"
    model.add_adapter(expert_name, config=config)

# Following Mixtral's approach to building a router
# Code from Mergoo's implementation, except this is sequence level not token level
class Router(nn.Module):
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

# Test
hidden_size = model.config.hidden_size
examples = ["Why is the sky blue?", "How do you make tea?", "What is quantum physics?"]
inputs = tokenizer(examples, return_tensors="pt", padding=True)
router = Router(hidden_size)
router(model.model.embed_tokens(inputs["input_ids"]))




