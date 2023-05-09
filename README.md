# symbolic_music_generation_transformers
## Project Members
Yumeng Zhang, Junyi Liu, Yifan Zhou
## Data
We use [GiantMIDI-Piano](https://github.com/bytedance/GiantMIDI-Piano) and [Pop909](https://github.com/music-x-lab/POP909-Dataset) as our training data. You can also access the dataset from the following Google Drive link:

GiantMIDI-Piano: https://drive.google.com/drive/folders/1-L9a69vldJ45tVy1hV964g9mEUKD58LJ?usp=sharing

Pop909: https://drive.google.com/drive/folders/1KSwmvu4SGJM2arzWiNDE3h1UEwk7koPI?usp=sharing

## Usage
GPT-2: To run the training and generation code, clone the reppository
```
git clone https://github.com/EEexplorer001/symbolic_music_generation_transformers.git
```
Then run the run.ipynb in the GPT_music folder. You can change the data path, token path, model save path and generation path to your own.

Theme transformer: run ThemeTransformer.ipynb in theme_transformer folder.
