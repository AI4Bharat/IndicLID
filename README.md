<div align="center">
	<h1><b><i>IndicLID</i></b></h1>
	<a href="https://ai4bharat.org/indiclid">Website</a> |
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
| Panjabi (Gurmukhi script) | pan_Guru |  
| Panjabi (Latin script) | pan_Latn |  
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
IndicLID is evaluated on [Bhasha-Abhijnaanam benchmark]() which is released alnog with this work. For native-script text, IndicLID has better language coverage than existing LIDs and is competitive or better than other LIDs. IndicLID model is 10 times faster and 4 times smaller than the [NLLB model]() also establish a strong baseline results on the roman-script text. For more details, refer our [paper]().

#### Native LID Results
Following table compares IndicLID-FTN model with the [NLLB model]() and the [CLD3 model](). We restrict the comparison to languages that are common with IndicLID (count of common languages is indicated in brackets). Throughput is number of sentence/second.

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
  - [Using hosted APIs](#using-hosted-apis)
  - [Accessing on ULCA](#accessing-on-ulca)
- [Running inference](#running-inference)
  - [Command line interface](#command-line-interface)
  - [Python interface](#python-interface)
- [Training model](#training-model)
  - [Setting up your environment](#setting-up-your-environment)
  - [Details of models and hyperparameters](#details-of-models-and-hyperparameters)
  - [Training procedure and code](#training-procedure-and-code)
  - [Evaluating trained model](#evaluating-trained-model)
  - [Detailed benchmarking results](#detailed-benchmarking-results)
- [Finetuning model on your data](#finetuning-model-on-your-data)
- [Directory structure](#directory-structure)
- [Citing](#citing)
  - [License](#license)
  - [Contributors](#contributors)
  - [Contact](#contact)
- [Acknowledgements]


## Resources

## Running Inference

## Training model

## Details of models and hyperparameters

## Directory structure


<!-- citing information -->

We would like to hear from you if:
- You are using our resources. Please let us know how you are putting these resources to use.
- You have any feedback on these resources.

<!-- License -->
### License
The IndicLID code (and models) are released under the MIT License.

<!-- Contributors -->
### Contributors
 - Yash Madhani <sub> ([AI4Bharat](https://ai4bharat.org), [IITM](https://www.iitm.ac.in)) </sub>
 - Anoop Kunchukuttan <sub> ([AI4Bharat](https://ai4bharat.org), [Microsoft](https://www.microsoft.com/en-in/)) </sub>
 - Mitesh M. Khapra <sub> ([AI4Bharat](https://ai4bharat.org), [IITM](https://www.iitm.ac.in)) </sub>

<!-- Contact -->
### Contact
- Anoop Kunchukuttan ([anoop.kunchukuttan@gmail.com](mailto:anoop.kunchukuttan@gmail.com))
- Mitesh Khapra ([miteshk@cse.iitm.ac.in](mailto:miteshk@cse.iitm.ac.in))
- Pratyush Kumar ([pratyush@cse.iitm.ac.in](mailto:pratyush@cse.iitm.ac.in))

## Acknowledgements

We would like to thank EkStep Foundation for their generous grant which helped in setting up the Centre for AI4Bharat at IIT Madras to support our students, research staff, data and computational requirements. We would like to thank The Ministry of Electronics and Information Technology (NLTM) for its grant to support the creation of datasets and models for Indian languages under its ambitious Bhashini project. We would also like to thank the Centre for Development of Advanced Computing, India (C-DAC) for providing access to the Param Siddhi supercomputer for training our models. Lastly, we would like to thank Microsoft for its grant to create datasets, tools and resources for Indian languages.
