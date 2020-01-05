
# Sentiment Analysis with Deep Learning

### A machine learning model that can provide insights by automatically analyzing product reviews and separating them into tags: Positive, Negative.

Customer Experience (CX) is the key to business success. In fact, 81% of marketers interviewed by Gartner said they expected their companies to compete mostly on the basis of CX in two years time, making CX the new marketing battlefront. Now, more than ever, it’s key for companies to pay close attention to Voice of Customer (VoC) to improve the customer experience. By analyzing and getting insights from customer feedback, companies have better information to make strategic decisions, an accurate understanding of what the customer actually wants and, as a result, a better experience for everyone.

This machine learning model can provide insights by automatically analyzing product reviews and separating theLoading and Pre-processingm into tags: Positive and Negative. By using sentiment analysis to structure product reviews, you can:

- Understand what your customers like and dislike about your product.
- Compare your product reviews with those of your competitors.
- Get the latest product insights in real-time, 24/7.
- Save hundreds of hours of manual data processing.


![sentiment](https://raw.githubusercontent.com/fcamuz/sentiment-analysis-with-deep-learning/master/images/sentiment.png)


### Dataset:
I use 'Amazon Product Reviews Dataset' from Kaggle. This dataset consists of total 4 millions Amazon customer reviews (input text) and star ratings (output labels). 

https://www.kaggle.com/bittlingmayer/amazonreviews


### Methodology:
CRISP-DM (Cross Industry Standard Process for Data Mining)
The process or methodology of CRISP-DM is described in these six major steps

#### 1.Business Understanding
Focuses on understanding the project objectives and requirements from a business perspective, and then converting this knowledge into a data mining problem definition and a preliminary plan.

#### 2.Data Understanding
Starts with an initial data collection and proceeds with activities in order to get familiar with the data, to identify data quality problems, to discover first insights into the data, or to detect interesting subsets to form hypotheses for hidden information.

#### 3.Data Preparation
The data preparation phase covers all activities to construct the final dataset from the initial raw data.

#### 4.Modeling
Modeling techniques are selected and applied.  Since some techniques like neural nets have specific requirements regarding the form of the data, there can be a loop back here to data prep.

#### 5.Evaluation
Once one or more models have been built that appear to have high quality based on whichever loss functions have been selected, these need to be tested to ensure they generalize against unseen data and that all key business issues have been sufficiently considered.  The end result is the selection of the champion model(s).

#### 6.Deployment
Generally this will mean deploying a code representation of the model into an operating system to score or categorize new unseen data as it arises and to create a mechanism for the use of that new information in the solution of the original business problem.  Importantly, the code representation must also include all the data prep steps leading up to modeling so that the model will treat new raw data in the same manner as during model development.

## Notebooks

There are 3 notebooks for this project. Since data is too large I chuncked the project into small pieces. Saved processed data, tokens, vectors and models regularly. 

- Loading and Pre-processing 

  [load_preprocess_data.ipynb](https://github.com/fcamuz/sentiment-analysis-with-deep-learning/blob/master/load_preprocess_data.ipynb)
   consists of functions and code to read the data, remove punctuations,  remove stopwords, lemmatize, tokenize and vectorize. There is also section for EDA. 

   CHRISP-DM phases¶

    Data Understanding and Data Preperation phases for CRISP-DM can be found in this noteboook.

   Table of Content:
  - 1.Import Libraries
  - 2.Define Functions
  - 3.Load-Read-Extract
  - 4.Pre-Processig
  - 5.Tokenizing-Sequenzing-Padding
  - 6.Exploratory Data Analysis (EDA)

  
- Modelling (Modeling and Evaluation)

  [model.ipynb](https://github.com/fcamuz/sentiment-analysis-with-deep-learning/blob/master/model.ipynb) has all base models train and test results. Tuning for best performing model is also available in this notebook.   

  CHRISP-DM phases¶

    Modelling and Evaluation phases for CRISP-DM can be found in this noteboook.
    
    Table of Content:
  - 1.Import Libraries
  - 2.Define Functions
  - 3.Modeling With Neural Networks
  - 4.Tuning the Best Model
  
- Prediction (Deployment)
  
  [prediction.ipynb ](https://github.com/fcamuz/sentiment-analysis-with-deep-learning/blob/master/prediction.ipynb) 
   consists the pipeline structure for scraping unseen data from Amazon website, pre-processing comments and predicting the labels. 

   CHRISP-DM phase

    Deployment phase for CRISP-DM can be found in this noteboook.
    
    Table of Content:
    - 1.Import Libraries
    - 2.Define Functions 
    - 3.Scraping Comments
    - 4.Pre-processing the New Data
    - 5.Prediction



  








References:

https://www.datasciencecentral.com/profiles/blogs/crisp-dm-a-standard-methodology-to-ensure-a-good-outcome

https://monkeylearn.com/blog/sentiment-analysis-of-product-reviews/