from moe import MoE, convert_linear_to_moe
from transformers import AutoTokenizer
from adapters import DoubleSeqBnConfig, AutoAdapterModel

model_name = "meta-llama/Llama-3.2-3B-Instruct"
model = AutoAdapterModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name, padding=True)
tokenizer.pad_token = tokenizer.eos_token # Just did this for now, should be changed to the actual pad token

model.config.router_layers = ["gate_proj", "up_proj", "down_proj", "q_proj", "k_proj", "v_proj", "o_proj"]
model.config.router_layers_index = list(range(model.config.num_hidden_layers))

# Houlby Adapter Config according to https://arxiv.org/pdf/1902.00751
adapter_config = DoubleSeqBnConfig()

# Create 8 experts
for i in range(8):
    expert_name = f"expert_{i}"
    model.add_adapter(expert_name, config=adapter_config)

# Alters the Llama adapter files to use the Router module.
for layer_ind, layer in enumerate(model.model.layers):
    config = model.config
    head_dim = config.hidden_size // config.num_attention_heads
    attn_bias = config.attention_bias

    # MLP
    layer.mlp.gate_proj = convert_linear_to_moe("gate_proj", config, layer_ind, config.hidden_size, config.intermediate_size, bias=False)
    layer.mlp.up_proj = convert_linear_to_moe("up_proj", config, layer_ind, config.hidden_size, config.intermediate_size, bias=False)
    layer.mlp.down_proj = convert_linear_to_moe("down_proj", config, layer_ind, config.intermediate_size, config.hidden_size, bias=False)

    # Attention
    layer.self_attn.q_proj = convert_linear_to_moe("q_proj", config, layer_ind, config.hidden_size, config.num_attention_heads * head_dim, bias=attn_bias)
    layer.self_attn.k_proj = convert_linear_to_moe("k_proj", config, layer_ind, config.hidden_size, config.num_key_value_heads * head_dim, bias=attn_bias)
    layer.self_attn.v_proj = convert_linear_to_moe("v_proj", config, layer_ind, config.hidden_size, config.num_key_value_heads * head_dim, bias=attn_bias)
    layer.self_attn.o_proj = convert_linear_to_moe("o_proj", config, layer_ind, config.hidden_size, config.hidden_size, bias=attn_bias)

# Test
hidden_size = model.config.hidden_size
examples = ["Why is the sky blue?", "How do you make tea?", "What is quantum physics?"]
inputs = tokenizer(examples, return_tensors="pt", padding=True)
outputs = model(**inputs)
print(outputs.logits.shape) # should be [3, seq_len, vocab_size]