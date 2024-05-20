<img src='images/hanzi_hero_icon.png' style="float: left; margin: 20px; height: 55px">


# Hanzi Hero: Empowering Kids to Master Chinese Characters
<div style="text-align: right"> <H3> By Jayme Zhang </div>


## Introduction to Problem Statement

While Chinese language is gaining popularity, its complex writing system presents a significant hurdle to fluency. Memorizing characters is essential for reading and pronunciation. Schools have been conducting regular Chinese spelling tests for students, aiming to build a strong foundation early on. However, this approach might prove challenging for Singaporean parents who juggle busy schedules and may not be proficient in Mandarin themselves.

A recent article by [CNA](https://www.channelnewsasia.com/singapore/census-2020-more-households-speaking-english-1966731) highlighted that in 2020, 48.3% of all the respondents said English was the language they used most frequently at home. For the Chinese ethnic group, the use of Mandarin as the predominant language fell from 47.7% in 2010 to 40.2% in 2020. The lack of exposure to the language both in school and at home may make it more difficult for students to get a good grasp of the language.

With these considerations in mind, this project aims to build an image recognition model for handwritten Chinese characters and deploy it to Streamlit, so that we can assist students and non-Chinese learners better with their learning journey.
  
This repository provides all the processes and codes needed to build the image recognition model.  A separate repository [Hanzi Hero Streamlit](https://github.com/jaymezhangjy/hanzi_hero_streamlit) focuses on deploying the trained model to a user interface using Streamlit Cloud. The user interface can be accessed via this [link](https://hanzi-hero.streamlit.app/): https://hanzi-hero.streamlit.app/.

## Problem Statement

### *How might we design a tool to automate the revision of Chinese spelling test for children, alleviating the current reliance on adult supervision?*

## Table of Contents

Our data pre-processing and modeling process are detailed in the following notebooks:

[01_EDA](01_EDA.ipynb)  
[02_Self_Trained_Model_Binary](02_Self_Trained_Model_Binary.ipynb)  
[03_Self_Trained_Model_Multiclass](03_Self_Trained_Model_Multiclass.ipynb)  
[04_Pre_Trained_Model_Multiclass](04_Pre_Trained_Model_Multiclass.ipynb)  
[05_SVM_Multiclass](05_SVM_Multiclass.ipynb)  
[06_Self_Trained_Model_Multiclass_HP_Tune](06_Self_Trained_Model_Multiclass_HP_Tune.ipynb)  

## Data Source

The data originates from the CASIA Offline database, a public resource developed by the Chinese National Laboratory of Pattern Recognition (NLPR) and the Institute of Automation of Chinese Academy of Sciences (CASIA). This dataset consists of handwritten samples contributed by 1,020 individuals between 2007 and 2010.  
  
The original custom-encoded binary files have been conveniently converted to PNG image files for ease of use in our model. Both the original files and PNG image files can be downloaded from [Kaggle](https://www.kaggle.com/datasets/pascalbliem/handwritten-chinese-character-hanzi-datasets/data).  
    
*This dataset from Kaggle is __14GB__ in size. [01_EDA](01_EDA.ipynb) is the only notebook which will require the entire dataset. From the next notebook onwards, we will use extracts from this data set, and the files are uploaded in GitHub.*

## Workflow Process

1. We started off with doing an exploratory data analysis on the above data source, and made use of the findings to assist us in our image pre-processing steps.

2. Next, we built a Convolution Neural Network (CNN) from scratch as a binary classifier for 1 Chinese characters. With a high accuracy of 92.11%, we were confident to use the same CNN to train a multiclass classifer for 10 Chinese characters. The accuracy score was 96.00%.

3. In addition to building self-trained models, we made use of transfer learning and implemented pre-trained models in the process, namely MobileNetV2 and ResNet50V2. The accuracy scores were 90.10% and 90.03% respectively.

4. Other than deep learning models, we also experimented with an image recognition model with the use of Support Vector Machine. The accuracy score was 83.29%.
  
    Here's a tabular comparison of all the metrics measured for all baseline models:

    | Model                   | Train loss | Val loss | Train accuracy | Val accuracy | Train AUC | Val AUC   |
    |-------------------------|------------|----------|----------------|--------------|-----------|-----------|
    | self_trained_multiclass | 0.144852   | 0.125192 | 0.953294       | 0.959972     | 0.997593  | 0.998727  |
    | pre_trained_mobilenetv2 | 0.138165   | 0.322845 | 0.970976       | 0.900983     | 0.999210  | 0.992990  |
    | pre_trained_resnet50v2  | 0.090398   | 0.319360 | 0.984487       | 0.900281     | 0.999851  | 0.992445  |
    | svm_multiclass          | 0.063645   | 0.831161 | 0.982319       | 0.832865     | 0.999691  | 0.977398  |

5. Based on the above comparison, we chose the self-trained multiclass model for hyperparameter tuning. We will be using KerasTuner to simplify and automate this process. After fitting our data into the best model and hyperparameters, the accuracy score improved from 96.00% to 97.82%.

    Here's a tabular comparison of all the metrics measured before and after hyperparameter tuning:

    | Model                               | Train loss | Val loss | Train accuracy | Val accuracy | Train AUC | Val AUC   |
    |-------------------------------------|------------|----------|----------------|--------------|-----------|-----------|
    | self_trained_multiclass             | 0.144852   | 0.125192 | 0.953294       | 0.959972     | 0.997593  | 0.998727  |
    | self_trained_multiclass_hp_tune     | 0.063176   | 0.066039 | 0.979316       | 0.978230     | 0.999179  | 0.998694  |

## Limitations

Currently, our model is trained on 10 Chinese characters. For the model to be adequately functional for the PSLE level, we would require a minimum of 1,200 Chinese characters. For a basic command of the language for daily usage, we would need the model to be trained on between 2,000 to 3,000 Chinese characters. This would be definitely be possible with additional computational capabilities.
  
We also noticed that the model is unable to provide the correct prediction when we deliberately added or missed a few strokes for the Chinese character. We would need to improve the precision of the model by training it on a larger dataset of Chinese characters with printed handwritings which are more similar to children's handwritings, rather than cursive ones. Moreover, the dataset that it was trained on was obtained from adults, and most would have developed their unique writing style by then.

While this repository focuses on the model itself, limitations and improvements pertaining to deployment onto Streamlit Cloud are detailed in the readme of [Hanzi Hero Streamlit](https://github.com/jaymezhangjy/hanzi_hero_streamlit) repository.

## Python Libraries Requirements

For the purpose of this project, installation of the following libraries is required:

1. `keras`
2. `keras-tuner`
3. `matplotlib`
4. `numpy`
5. `os`
6. `pandas`
7. `scikit-learn` or `sklearn`
8. `skimage`
9. `tensorflow`
10. `time`

## Downloads from External Links

1. As mentioned above, for [01_EDA](01_EDA.ipynb), the original dataset can be downloaded from [Kaggle](https://www.kaggle.com/datasets/pascalbliem/handwritten-chinese-character-hanzi-datasets/data).
2. For [06_Self_Trained_Model_Multiclass_HP_Tune](06_Self_Trained_Model_Multiclass_HP_Tune.ipynb), search results from Kera Tuner are saved in a separate folder and it can be downloaded from this [link](https://drive.google.com/drive/folders/1mb_Ejpj8TKNf_SxF6wMkzH2PUmvuOxUa?usp=sharing).