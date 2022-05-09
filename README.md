# Adima

This is an official pytorch implementation of our ICASSP 2022 paper [ADIMA: ABUSE DETECTION IN MULTILINGUAL AUDIO](https://arxiv.org/pdf/2202.07991.pdf). In this repository, we provide the dataset and codebase for exploring our work. 

Please visit the [Project Page](https://sharechat.com/research/adima) for more details about the dataset. 

If you find ADIMA useful in your research, please use the following BibTeX entry for citation.

```
@inproceedings{gupta2022adima,
  title={ADIMA: Abuse Detection In Multilingual Audio},
  author={Gupta, Vikram and Sharon, Rini and Sawhney, Ramit and Mukherjee, Debdoot},
  booktitle={ICASSP 2022-2022 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={6172--6176},
  year={2022},
  organization={IEEE}
}
```

# ADIMA Dataset

Audio files, annotations and code of ADIMA can be downloaded from [here](https://drive.google.com/drive/folders/1geQ4PlXGsNCvPQDT3tKztvAu817PB5TP).

The annotation folder contains the train and test splits used in our paper for all the languages. Each row of the csv contains the name of audio file and ground-truth indicating the presence of profanity.

For example, “Hindi_train.csv” and “Hindi_test.csv” contains the train and test data for Hindi language. 

Each row of this file has the name of audio file and its label. Labels “Yes” and “No” represent the presence and absence of profanity in the audio file.
