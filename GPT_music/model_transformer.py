from transformers import GPT2LMHeadModel, GPT2Config
config = GPT2Config(
    n_positions=2048,
    n_embd=512,
    n_layer=4,
    n_head=4,
    n_inner=2048,
    vocab_size=500, # Length of the vocab
    padding_token_id=0, # Padding token id
    bos_token_id=1, # BOS token id
    eos_token_id=2, # EOS token id
)
model = GPT2LMHeadModel(config)
# model = GPT2LMHeadModel.from_pretrained('gpt2-small', config=config, use_auth_token=True)