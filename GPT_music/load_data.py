# This code is modified from the original code base: https://github.com/Natooz/MidiTok
# Original Author: 2021 Nathan Fradet


from pathlib import Path
import json

import torch
from torch import Tensor, LongTensor, stack, flip, cat, full, argmax
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader

from transformers.data.data_collator import DataCollatorMixin, DataCollatorForLanguageModeling

from miditoolkit import MidiFile
from tqdm import tqdm

import os
from unidecode import unidecode
from sklearn.model_selection import train_test_split

def make_meta_data(root):
    '''
    To load data from giantMIDI piano.
    Due to limited computing resources, we chose music piece from only two composers, Bach JohannSebastian and
    Chopin Frederic. There's a total of 249 midis in the dataset.
    '''
    columns = ['path', 'name0', 'name1', 'class']
    meta_dict = {col: [] for col in columns}
    class_num = 0

    midi_file_list = sorted(os.listdir(root))

    for midi_file in midi_file_list:
        name0, name1 = unidecode(midi_file.split(',')[0]).replace(" ", ""), unidecode(midi_file.split(',')[1]).replace(" ", "")
        if (name0 == 'Bach' and name1 == 'JohannSebastian') or (name0 == 'Chopin' and name1 == 'Frederic'):
            if name0 not in meta_dict['name0'] or name1 not in meta_dict['name1']:               
                new_row = {'path': os.path.join(root, midi_file),
                          'name0': name0,
                          'name1': name1,
                          'class': class_num,}
                class_num += 1
            else:
                new_row = {'path': os.path.join(root, midi_file),
                          'name0': name0,
                          'name1': name1,
                          'class': meta_dict['class'][-1],}

            [meta_dict[key].append(new_row[key]) for key in meta_dict.keys()]

    return meta_dict

def make_meta_data_pop(root):
    '''
    load Pop909 dataset
    '''
    columns = ['path', 'class']
    meta_dict = {col: [] for col in columns}
    length = len(os.listdir(root))
    
    for i in range(1, length):
        new_row = {'path': os.path.join(root, str(i).zfill(3), str(i).zfill(3) + '.mid'),
                  'class': str(i),}
        
        [meta_dict[key].append(new_row[key]) for key in meta_dict.keys()]
        
    return meta_dict
    
class MIDIDataset(Dataset):
    """
    Construct MIDI dataset from data path
    """

    def __init__(self, files_paths, min_seq_len, max_seq_len, tokenizer=None):
        samples = []
        file_labels = []

        for file_path in tqdm(files_paths, desc=f'Loading data: {files_paths[0].parent}'):
            if file_path.suffix in ["mid", "midi", "MID", "MIDI"]:
                midi = MidiFile(file_path)
                for _ in range(len(midi.instruments) - 1):
                    del midi.instruments[1]  # removes all tracks except first one
                tokens = tokenizer.midi_to_tokens(midi)[0].ids
            else:
                with open(file_path) as json_file:
                    tokens = json.load(json_file)['ids'][0]  # first track
            i = 0
            while i < len(tokens):
                if i >= len(tokens) - min_seq_len:
                    break  # last sample is too short
                samples.append(torch.LongTensor(tokens[i:i + max_seq_len]))
                file_labels.append(os.path.basename(file_path)[:-5] + f'_startat{i}')
                i += len(samples[-1])  # could be replaced with max_seq_len

        self.samples = samples
        self.file_labels = file_labels

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return {"input_ids": self.samples[idx], "labels": self.samples[idx], "file_labels": self.file_labels[idx]}
    
    
class DataCollatorGen(DataCollatorForLanguageModeling):
    def __init__(self, pad_token, return_tensors="pt"):

        self.pad_token = pad_token
        self.return_tensors = return_tensors

    def __call__(self, batch, return_tensors=None):
        x, y = self._pad_batch(batch, self.pad_token), self._pad_batch(batch, -100)
        return {"input_ids": x, "labels": y} 

    def _pad_batch(self, examples, pad_token):
        """
        To pad the sequence that is not the same length as max length
        """

        length_of_first = examples[0]["input_ids"].size(0)

        are_tensors_same_length = all(x["input_ids"].size(0) == length_of_first for x in examples)
        if are_tensors_same_length:
            return torch.stack([e["input_ids"] for e in examples], dim=0).long()

        return pad_sequence([e["input_ids"] for e in examples], batch_first=True, padding_value=pad_token).long()

