# This code is modified from the original code base: https://github.com/Natooz/MidiTok
# Original Author: 2021 Nathan Fradet


from transformers import GenerationConfig
from copy import deepcopy
from tqdm import tqdm
from torch import flip, cat, full
from torch.nn.utils.rnn import pad_sequence

def collate_gen_left(batch):
    # Collate the testing data and pad the sequence in the left.
    bos_shape = (1,)
    file_labels = [seq["file_labels"] for seq in batch]
    batch = [flip(cat([full(bos_shape, 1), seq["input_ids"]], dim=0), dims=(0,)) for seq in batch]
    batch = pad_sequence(batch, batch_first=True, padding_value=0)  # (N,T) or (N,T,Z)
    batch = flip(batch, dims=(1,)).long()
    return batch, file_labels
    
generation_config = GenerationConfig(
    max_new_tokens=512, # Output length: 512
    num_beams=1,   
    do_sample=True, 
    temperature=0.9, # change variation
    top_k=5,
    top_p=0.95,
    epsilon_cutoff=3e-4,
    eta_cutoff=1e-3,
    pad_token_id=0, # 0 for padding token id
)

def generate_batch(dataloader_test, model, device, tokenizer, gen_results_path):
    model.eval()
    for batch, file_labels in tqdm(dataloader_test, desc='Testing model / Generating results'):  
    
        res = model.generate(batch.to(model.device), generation_config=generation_config) 
    
        # Saves the generated music, as MIDI files and tokens (json)
        for prompt, continuation, file_label in zip(batch, res, file_labels):
            generated = continuation[len(prompt):]
            tokens = [generated, prompt, continuation]  # list compr. as seqs of dif. lengths
            tokens = [seq.tolist() for seq in tokens]
            continuation = [continuation.tolist()]
            prompt = [prompt.tolist()]
            generated = [generated.tolist()]
    
            midi_continuation = tokenizer.tokens_to_midi(deepcopy(continuation), time_division=384)
            midi_original = tokenizer.tokens_to_midi(deepcopy(prompt), time_division=384)
            midi_generated = tokenizer.tokens_to_midi(deepcopy(generated), time_division=384)
    
            midi_continuation.dump(gen_results_path + 'continuation_' + file_label + '.mid')
            midi_original.dump(gen_results_path + 'original_' + file_label + '.mid')
            midi_generated.dump(gen_results_path + 'generated_' + file_label + '.mid')
            tokenizer.save_tokens(tokens, gen_results_path + file_label + '.json')
