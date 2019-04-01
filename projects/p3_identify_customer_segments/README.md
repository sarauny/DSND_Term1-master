# Data Scientist Nanodegree
# Unsupervised Learning
## Project: Identify Customer Segments

### Project Overview
In this project, you will work with real-life data provided to us by our Bertelsmann partners AZ Direct and Arvato Finance Solution. The data here concerns a company that performs mail-order sales in Germany. Their main question of interest is to identify facets of the population that are most likely to be purchasers of their products for a mailout campaign. Your job as a data scientist will be to use unsupervised learning techniques to organize the general population into clusters, then use those clusters to see which of them comprise the main user base for the company. Prior to applying the machine learning methods, you will also need to assess and clean the data in order to convert the data into a usable form.

### Install
It is highly recommended that you use the project workspace to complete and submit this project. Should you decide to download the data, you will need to remove it from your computer via the agreement with Arvato Bertlesmann. If you choose to use your machine to complete this project, below are a list of the requirements.

This project uses Python 3 and is designed to be completed through the Jupyter Notebooks IDE. It is highly recommended that you use the Anaconda distribution to install Python, since the distribution includes all necessary Python libraries as well as Jupyter Notebooks. The following libraries are expected to be used in this project:

- NumPy
- pandas
- Sklearn / scikit-learn
- Matplotlib (for data visualization)
- Seaborn (for data visualization)

### Data
The files available for the project are:

- Udacity_AZDIAS_Subset.csv: Demographic data for the general population of Germany; 891211 persons (rows) x 85 features (columns).
- Udacity_CUSTOMERS_Subset.csv: Demographic data for customers of a mail-order company; 191652 persons (rows) x 85 features (columns).
- Data_Dictionary.md: Information file about the features in the provided datasets.
- AZDIAS_Feature_Summary.csv: Summary of feature attributes for demographic data.
- Identify_Customer_Segments.ipynb: Jupyter Notebook divided into sections and guidelines for completing the project. The notebook provides more details and tips than the outline given here.

Based on the agreement with Arvato Bartlesmann, data is not available to be shared online.

### Project Details

### Step 1: Preprocessing
This step includes:
 - detectioon and processing of missing or unknown data
 - re-encoding of non-numeric features
 - removing unnecessary features for later clustering work

### Step 2: Feature Transformation
Feature scaling and PCA dimensionality reduction are executed at this step.

### Step 3: Clustering
K-means method is used for this step.
Best n_clusters are selected based on the average distance from centroids to samples.

### Step 4: Review and Submit the Project
Submitted and reviewed files are:

- Identify_Customer_Segments.ipynb
- Identify_Customer_Segments.html