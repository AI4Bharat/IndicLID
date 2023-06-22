<div align="center">
	<h1><b><i>IndicLID</i></b></h1>
	<a href="https://ai4bharat.iitm.ac.in/indiclid">Website</a> |
	<a href="#download-indiclid-model">Downloads</a> |
	<a href="">Demo</a>
  <br><br>
</div>


<!-- description about IndicLID -->
***IndicLID***, is a language identifier for ***all 22 Indian languages*** listed in the Indian constitution in both native-script and romanized text. IndicLID is the ***first LID for romanized text in Indian languages***. It is a two stage classifier that is ensemble of a fast linear classifier and a slower classifier finetuned from a pre-trained LM. It can ***predict 47 classes (24 native-script classes and 21 roman-script classes plus English and Others)***. All the classes are listed below. 


<!-- list of languages IndicLID supports -->
## Languages Supported
| Language | IndicLID Code | 
|----------|---------------|
| Assamese (Bengali script) | asm_Beng |  
| Assamese (Latin script) | asm_Latn |  
| Bangla (Bengali script) | ben_Beng |  
| Bangla (Latin script) | ben_Latn |  
| Bodo (Devanagari script) | brx_Deva |  
| Bodo (Latin script) | brx_Latn |  
| Dogri (Devanagari script) | doi_Deva |  
| Dogri (Latin script) | doi_Latn | 
| English (Latin script) | eng_Latn |  
| Gujarati (Gujarati script) | guj_Gujr |  
| Gujarati (Latin script) | guj_Latn |  
| Hindi (Devanagari script) | hin_Deva |  
| Hindi (Latin script) | hin_Latn |  
| Kannada (Kannada script) | kan_Knda |  
| Kannada (Latin script) | kan_Latn |  
| Kashmiri (Perso_Arabic script) | kas_Arab |  
| Kashmiri (Devanagari script) | kas_Deva |  
| Kashmiri (Latin script) | kas_Latn |  
| Konkani (Devanagari script) | kok_Deva |  
| Konkani (Latin script) | kok_Latn |  
| Maithili (Devanagari script) | mai_Deva |  
| Maithili (Latin script) | mai_Latn |  
| Malayalam (Malayalam script) | mal_Mlym |  
| Malayalam (Latin script) | mal_Latn |  
| Manipuri (Bengali script) | mni_Beng |  
| Manipuri (Meetei_Mayek script) | mni_Meti |  
| Manipuri (Latin script) | mni_Latn |  
| Marathi (Devanagari script) | mar_Deva |  
| Marathi (Latin script) | mar_Latn |  
| Nepali (Devanagari script) | nep_Deva |  
| Nepali (Latin script) | nep_Latn |  
| Oriya (Oriya script) | ori_Orya |  
| Oriya (Latin script) | ori_Latn |  
| Punjabi (Gurmukhi script) | pan_Guru |  
| Punjabi (Latin script) | pan_Latn |  
| Sanskrit (Devanagari script) | san_Deva |  
| Sanskrit (Latin script) | san_Latn |  
| Santali (Ol_Chiki  script) | sat_Olch |  
| Sindhi (Perso_Arabic script) | snd_Arab |  
| Sindhi (Latin script) | snd_Latn |  
| Tamil (Tamil script) | tam_Tamil |  
| Tamil (Latin script) | tam_Latn |  
| Telugu (Telugu script) | tel_Telu |  
| Telugu (Latin script) | tel_Latn |  
| Urdu (Perso_Arabic script) | urd_Arab |  
| Urdu (Latin script) | urd_Latn |  
| Other | other |

### Evaluation Results
IndicLID is evaluated on [Bhasha-Abhijnaanam benchmark](https://huggingface.co/datasets/ai4bharat/Bhasha-Abhijnaanam) which is released alnog with this work. For native-script text, IndicLID has better language coverage than existing LIDs and is competitive or better than other LIDs. IndicLID model is 10 times faster and 4 times smaller than the [NLLB model](https://huggingface.co/docs/transformers/model_doc/nllb) also establish a strong baseline results on the roman-script text. For more details, refer our [paper](https://arxiv.org/abs/2305.15814).

#### Native LID Results
Following table compares IndicLID-FTN model with the [NLLB model](https://huggingface.co/docs/transformers/model_doc/nllb) and the [CLD3 model](https://github.com/google/cld3). We restrict the comparison to languages that are common with IndicLID (count of common languages is indicated in brackets). Throughput is number of sentence/second.

| Model | Precison | Recall | F1-score | Accuracy | Throughput | Model Size |  
| ----- | -------- | ------ | -------- | -------- | ---------- | ---------- |  
| IndicLID-FTN-8-dim (24) | 0.98 | 0.99 | 0.98 | 0.98 | 30,303 | 318M |
|  |  |  |  |  |  |  |  
| IndicLID-FTN-4-dim (12) | 0.99 | 0.98 | 0.99 | 0.98 | 47,619 | 208M |
| IndicLID-FTN-8-dim (12) | 1.00 | 0.99 | 0.99 | 0.98 | 33,333 | 318M |
| CLD3 (12) | 0.99 | 0.98 | 0.98 | 0.98 | 4,861 | -  |
|  |  |  |  |  |  |  |
| IndicLID-FTN-4-dim (20) | 0.98 | 0.98 | 0.98 | 0.98 | 41,666 | 208M |
| IndicLID-FTN-8-dim (20) | 0.98 | 0.99 | 0.98 | 0.98 | 29,411 | 318M |
| NLLB (20) | 0.99 | 0.99 | 0.99 | 0.98 | 4,970 | 1.1G |



#### Roman LID Results
Following table presents the results of different model variants on the romanized testset. Throughput is number of sentence/second.

| Model | Precison | Recall | F1-score | Accuracy | Throughput | Model Size |  
| ----- | -------- | ------ | -------- | -------- | ---------- | ---------- |  
| IndicLID-FTR (dim-8) | 0.63 | 0.78 | 0.63 | 0.71 | 37,037 | 357 M |
| IndicLID-BERT (unfeeze-layer-1) | 0.73 | 0.84 | 0.75 | 0.80 | 3 | 1.1 GB |
| IndicLID (threshold-0.6) | 0.73 | 0.85 | 0.75 | 0.80 | 10 | 1.4 GB |

<!-- index with hyperlinks (Table of contents) -->

- [Table of contents](#table-of-contents)
- [Resources](#resources)
  - [Download IndicLID model](#download-indiclid-model)
- [Running inference](#running-inference)
  - [Interface](#interface)
- [Training model](#training-model)
  - [Setting up your environment](#setting-up-your-environment)
  - [Training procedure and code](#training-procedure-and-code)
  - [Evaluating trained model](#evaluating-trained-model)
- [Directory structure](#directory-structure)
- [Citing information](#citing-information)
  - [License](#license)
  - [Contributors](#contributors)
  - [Contact](#contact)
- [Acknowledgements](#acknowledgements)


## Resources

### Download Bhasha-Abhijnaanam Test Set
[Huggingface](https://huggingface.co/datasets/ai4bharat/Bhasha-Abhijnaanam)

### Download Training Data
[Native Script Training Data](https://github.com/AI4Bharat/IndicLID/releases/download/v1.0/native_script_train_valid_data.zip)

[Roman Script Training Data](https://github.com/AI4Bharat/IndicLID/releases/download/v1.0/roman_script_train_valid_data.zip)


### Download IndicLID model
<!-- hyperlinks for downloading the models -->
IndicLID-FTN [v1.0](https://github.com/AI4Bharat/IndicLID/releases/download/v1.0/indiclid-ftn.zip)

IndicLID-FTR [v1.0](https://github.com/AI4Bharat/IndicLID/releases/download/v1.0/indiclid-ftr.zip)

IndicLID-BERT [v1.0](https://github.com/AI4Bharat/IndicLID/releases/download/v1.0/indiclid-bert.zip)

<!-- mirror links set up the public drive -->	


## Running Inference

### Interface
<!-- colab integratation on running the model on custom input python script -->
Inference Notebook --> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1pLMeaGhYgfNRmYHPHkvAmcR-8xMMZOme?usp=sharing)



## Training model
###  Setting up your environment
<details><summary> Click to expand </summary>

```bash
pip3 install fasttext
pip3 install transformers
```
</details>

### Training procedure and code

We train 3 models separately which are the components of IndicLID model. Please refer to the paper for more architectural detials.

We use fasttext models to train out IndicLID-FTR and IndicLID-FTN component. Following are the steps to train our fasttext models.
- Create a train file that contains the train sentences in the following format,
"__label__langcode <space> <Train_Sentence>"
- following is the script to train the fasttext model
```
import fasttext
import csv
import sys
	
model = fasttext.train_supervised(
    input = '../corpus/train_combine.txt', 
    loss = 'hs',
    verbose = 1,
    dim = 8,
    autotuneValidationFile = '../corpus/valid_combine.txt', 
    autotuneDuration = 14400*3
    )
model.save_model("../result/model_baseline_roman.bin")
```
	
For our IndicLID-BERT model, we finetune [IndicBERT](https://github.com/AI4Bharat/IndicBERT) model with our romaanized training data. Script for the training IndiLID-BERT model can be found [here](https://github.com/AI4Bharat/IndicLID/blob/master/final_runs_train/roman_model/finetuning/IndicBERT/unfreeze_layers/train.py).

	
### Evaluating trained model
Script to generate the output can be found [here](https://github.com/AI4Bharat/IndicLID/blob/master/Inference/ai4bharat/IndicLID.py).


## Directory structure
```
IndicLID/
├── Benchmark
│   ├── compile_final_pilot_1.py
│   ├── create_benchmark_extra.py
│   └── create_benchmark.py
├── deployement
│   ├── test_script.py
│   └── working
│       └── IndicLID.py
├── filter_Dakshina
│   ├── merge_final_native.py
│   ├── merge_final.py
│   └── separate_validation_set.py
├── final_runs_ACL_inference
│   ├── analysis
│   │   ├── language_wise
│   │   │   ├── inference.py
│   │   │   ├── prepare_corpus.py
│   │   │   ├── train_fasttext.py
│   │   │   ├── train_IndicBERT.py
│   │   │   └── word_overlap_confustion_matrix.py
│   │   ├── length_wise
│   │   │   ├── acc_len_wise_analysis.py
│   │   │   └── save_prediction_dict.py
│   │   └── word_embeddings
│   │       ├── PCA_cluster_embeddings.py
│   │       ├── TSNE_cluster_embeddings.py
│   │       └── word_neighbours.py
│   ├── native_model
│   │   ├── cld3_comparison
│   │   │   ├── cld3
│   │   │   │   ├── inference.py
│   │   │   │   ├── inference_time_1.py
│   │   │   │   ├── inference_time.py
│   │   │   │   └── prepare_corpus.py
│   │   │   ├── fasttext_4
│   │   │   │   ├── inference.py
│   │   │   │   ├── inference_time_1.py
│   │   │   │   ├── inference_time.py
│   │   │   │   └── prepare_corpus.py
│   │   │   └── fasttext_8
│   │   │       ├── inference.py
│   │   │       ├── inference_time_1.py
│   │   │       ├── inference_time.py
│   │   │       └── prepare_corpus.py
│   │   ├── corpus_inf_native
│   │   │   ├── lang_stat_test_native.csv
│   │   │   └── prepare_corpus.py
│   │   ├── fasttext
│   │   │   └── tune_run
│   │   │       ├── inference.py
│   │   │       ├── inference_time_1.py
│   │   │       ├── inference_time.py
│   │   │       ├── post_error_analysis.py
│   │   │       ├── prepare_corpus.py
│   │   │       ├── temp.sh
│   │   │       └── train.py
│   │   ├── finetuning
│   │   │   ├── IndicBERT
│   │   │   │   ├── freezed_bert_all_layer
│   │   │   │   │   ├── inference.py
│   │   │   │   │   ├── len_wise_analysis.py
│   │   │   │   │   └── train.py
│   │   │   │   └── unfreeze_layers
│   │   │   │       ├── inference.py
│   │   │   │       ├── len_wise_analysis.py
│   │   │   │       ├── temp.sh
│   │   │   │       └── train.py
│   │   │   ├── MuRIL
│   │   │   │   ├── freezed_bert_all_layer
│   │   │   │   │   ├── inference.py
│   │   │   │   │   └── train.py
│   │   │   │   └── unfreeze_layers
│   │   │   │       ├── inference.py
│   │   │   │       ├── len_wise_analysis.py
│   │   │   │       ├── temp.sh
│   │   │   │       └── train.py
│   │   │   └── XMLR
│   │   │       └── freezed_bert_all_layer
│   │   │           ├── inference.py
│   │   │           └── train.py
│   │   └── nllb_comparison
│   │       ├── fasttext_4
│   │       │   ├── inference.py
│   │       │   ├── inference_time_1.py
│   │       │   ├── inference_time.py
│   │       │   └── prepare_corpus.py
│   │       ├── fasttext_8
│   │       │   ├── inference.py
│   │       │   ├── inference_time_1.py
│   │       │   ├── inference_time.py
│   │       │   └── prepare_corpus.py
│   │       ├── indicbert
│   │       │   ├── inference.py
│   │       │   └── prepare_corpus.py
│   │       ├── indiclid_fast_4
│   │       │   ├── 2_stage_inference.py
│   │       │   └── prepare_corpus.py
│   │       ├── indiclid_fast_8
│   │       │   ├── 2_stage_inference.py
│   │       │   └── prepare_corpus.py
│   │       └── nllb
│   │           ├── inference.py
│   │           ├── inference_time_1.py
│   │           ├── inference_time.py
│   │           └── prepare_corpus.py
│   ├── roman_model
│   │   ├── corpus_inf_roman
│   │   │   ├── lang_stat_test_roman.csv
│   │   │   ├── lang_stat_test_romanized_indicxlit.csv
│   │   │   └── prepapre_test_set.py
│   │   ├── fasttext
│   │   │   └── tune_run
│   │   │       ├── inference.py
│   │   │       ├── inference_time_1.py
│   │   │       └── inference_time.py
│   │   └── finetuning
│   │       ├── IndicBERT
│   │       │   ├── freezed_bert_all_layer
│   │       │   │   ├── inference.py
│   │       │   │   ├── len_wise_analysis.py
│   │       │   │   └── train.py
│   │       │   └── unfreeze_layers
│   │       │       ├── inference.py
│   │       │       ├── inference_time_1.py
│   │       │       ├── inference_time.py
│   │       │       ├── len_wise_analysis.py
│   │       │       ├── temp.sh
│   │       │       └── train.py
│   │       ├── MuRIL
│   │       │   ├── freezed_bert_all_layer
│   │       │   │   ├── inference.py
│   │       │   │   ├── slurm-120303.out
│   │       │   │   └── train.py
│   │       │   └── unfreeze_layers
│   │       │       ├── inference.py
│   │       │       ├── len_wise_analysis.py
│   │       │       ├── temp.sh
│   │       │       └── train.py
│   │       └── XMLR
│   │           └── freezed_bert_all_layer
│   │               ├── inference.py
│   │               └── train.py
│   ├── two_stage
│   │   ├── IndicBERT
│   │   │   ├── 2_stage_inference.py
│   │   │   ├── display_confusion_matrix.py
│   │   │   ├── inference_time_1.py
│   │   │   ├── inference_time.py
│   │   │   └── prepare_scored_dakshina_romanized.py
│   │   └── MuRIL
│   │       └── 2_stage_inference.py
│   └── two_stage_native
│       └── IndicBERT_fasttext_8
│           ├── 2_stage_inference.py
│           ├── display_confusion_matrix.py
│           └── prepare_scored_dakshina_romanized.py
├── nueral_net
│   ├── experiments
│   │   ├── skeleton
│   │   │   ├── create_sen_embed.py
│   │   │   ├── inference.py
│   │   │   ├── prepare_corpus.py
│   │   │   └── train.py
│   │   └── skeleton_transform
│   │       ├── inference.py
│   │       ├── prepare_corpus.py
│   │       └── train.py
│   └── experiments_tune
│       └── skeleton_tuning
│           ├── inference.py
│           ├── prepare_corpus.py
│           └── train.py
├── preprocess_indiccorp
│   ├── IndicCorp_data
│   │   ├── extract_and_combine.log
│   │   ├── extract_and_combine.py
│   │   ├── indiccorp_combine_stats.csv
│   │   └── readme.md
│   ├── IndicCorp_data_subset
│   │   ├── combine_sources.py
│   │   ├── indiccorp_combine_stats.csv
│   │   └── readme.md
│   ├── IndicCorp_data_subset_tok_norm
│   │   ├── readme.md
│   │   └── tokenize_and_normalize.py
│   ├── IndicCorp_data_subset_tok_norm_romanized
│   │   ├── create_ip_word_list.py
│   │   ├── fairseq_postprocess.py
│   │   ├── indic_tok.py
│   │   ├── interactive.sh
│   │   ├── lang_list.txt
│   │   ├── preprocess_en.py
│   │   ├── readme.md
│   │   └── romanize_corpus.py
│   └── IndicCorp_data_subset_tok_norm_romanized_100k_sample_cleaned
│       ├── create_ip_word_list.py
│       ├── fairseq_postprocess.py
│       ├── indic_tok.py
│       ├── interactive.sh
│       ├── lang_list.txt
│       ├── preprocess_en.py
│       └── romanize_corpus.py
└── README.md
```

<!-- Citing information -->

We would like to hear from you if:
- You are using our resources. Please let us know how you are putting these resources to use.
- You have any feedback on these resources.

<!-- License -->
### License
The IndicLID code (and models) are released under the MIT License.

<!-- Contributors -->
### Contributors
 - Yash Madhani <sub> ([AI4Bharat](https://ai4bharat.iitm.ac.in/), [IITM](https://www.iitm.ac.in)) </sub>
 - Mitesh M. Khapra <sub> ([AI4Bharat](https://ai4bharat.iitm.ac.in/), [IITM](https://www.iitm.ac.in)) </sub>
 - Anoop Kunchukuttan <sub> ([AI4Bharat](https://ai4bharat.iitm.ac.in/), [Microsoft](https://www.microsoft.com/en-in/), [IITM](https://www.iitm.ac.in)) </sub>
	
<!-- Contact -->
### Contact
- Yash Madhani <sub> ([AI4Bharat](https://ai4bharat.iitm.ac.in/), [IITM](https://www.iitm.ac.in)) </sub>	
- Anoop Kunchukuttan ([anoop.kunchukuttan@gmail.com](mailto:anoop.kunchukuttan@gmail.com))
- Mitesh Khapra ([miteshk@cse.iitm.ac.in](mailto:miteshk@cse.iitm.ac.in))

## Acknowledgements
	
We would like to thank the Ministry of Electronics and Information Technology of the Government of India for their generous grant through the Digital India Bhashini project. We also thank the Centre for Development of Advanced Computing for providing compute time on the Param Siddhi Supercomputer. We also thank Nilekani Philanthropies for their generous grant towards building datasets, models, tools and resources for Indic languages. We also thank Microsoft for their grant to support research on Indic languages. We would like to thank Jay Gala and Ishvinder Sethi for their help in coordinating the annotation work. Most importantly we would like to thank all the annotators who helped create the Bhasha-Abhijnaanam benchmark.
