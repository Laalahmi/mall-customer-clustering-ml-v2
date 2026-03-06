Mall Customer Segmentation – Machine Learning Project

Customer segmentation project developed for CST2216 – Modularizing and Deploying Machine Learning Code at Algonquin College.

This project implements an unsupervised machine learning pipeline using KMeans clustering to group mall customers into meaningful behavioral segments based on their demographic and spending characteristics.

The application is deployed as an interactive Streamlit dashboard where users can explore cluster insights and predict the segment of a new customer.

Project Overview

Customer segmentation is a common business analytics task used to better understand different groups of customers and tailor marketing strategies accordingly.

In this project, we apply KMeans clustering to segment customers based on:

Age

Annual Income

Spending Score

The pipeline automatically searches for the optimal number of clusters (K) using Silhouette Score evaluation.

The trained model is then deployed through a Streamlit web application that allows interactive exploration of the clustering results.

Key Features

Modular machine learning pipeline

Data preprocessing and feature scaling

Automatic optimal K selection

Cluster profiling and interpretation

Interactive Streamlit dashboard

Predict customer segment from user input

Professional logging and error handling

Model artifact saving using joblib

Clean modular project architecture

Technologies Used

Python libraries used in this project include:

Python

pandas

numpy

scikit-learn

joblib

Streamlit

matplotlib

seaborn

Project Structure
project_3_unsupervised_clustering
│
├── assets
│   └── algonquin_logo.png
│
├── data
│   └── mall_customers.csv
│
├── models
│   └── clustering_bundle.joblib
│
├── src
│   ├── config.py
│   ├── logger.py
│   ├── data_loader.py
│   ├── preprocess.py
│   ├── clustering.py
│   ├── artifacts.py
│   └── utils.py
│
├── app.py
├── train.py
├── requirements.txt
└── README.md
Machine Learning Pipeline

The pipeline follows these steps:

1. Data Loading

The dataset is loaded and validated to ensure the required columns exist.

2. Data Cleaning

Duplicates are removed and numeric columns are validated.

3. Feature Selection

The clustering model uses the following features:

Age

Annual Income

Spending Score

4. Feature Scaling

Features are scaled using StandardScaler to normalize distances for KMeans.

5. KMeans Training

Multiple values of K (2–10) are tested.

Each candidate model is evaluated using:

Inertia

Silhouette Score

6. Best Model Selection

The model with the highest silhouette score is selected.

7. Cluster Profiling

Average values for each cluster are computed to interpret customer segments.

8. Model Artifact Saving

The trained model, scaler, and cluster summaries are saved as a deployment bundle.

Streamlit Application

The Streamlit dashboard includes three main sections:

Overview

Explains the project and shows clustering evaluation results.

Cluster Insights

Displays cluster profiles and interpretation of customer segments.

Predict Segment

Allows users to input customer characteristics and predict their cluster.

Running the Project Locally
1. Clone the repository
git clone https://github.com/YOUR_USERNAME/mall-customer-clustering-ml.git
cd mall-customer-clustering-ml
2. Create virtual environment
python -m venv .venv

Activate environment:

Windows

.venv\Scripts\activate

Mac/Linux

source .venv/bin/activate
3. Install dependencies
pip install -r requirements.txt
4. Train the clustering model
python train.py

This step will generate:

models/clustering_bundle.joblib
5. Launch the Streamlit application
streamlit run app.py

The dashboard will open in your browser.

Deployment

The application is designed to be deployed on:

Streamlit Community Cloud

Deployment steps include:

Push repository to GitHub

Connect repository to Streamlit Cloud

Set app.py as the main file

Streamlit installs dependencies automatically using requirements.txt

Academic Context

This project was developed as part of the course:

CST2216 – Modularizing and Deploying Machine Learning Code

Algonquin College
School of Advanced Technology

Credits

Student:
Mohammed Laalahmi

Professor:
Dr. Umer Altaf

Algonquin College