# Healthcare Fraud Detection System

## Overview

Healthcare fraud represents a significant and ongoing challenge, leading to billions of dollars in losses annually and diverting resources from legitimate patient care. This project provides a comprehensive, data-driven solution for detecting and preventing fraudulent healthcare claims using machine learning and real-time analytics.

The dataset has been fetched from kaggle's healthcare fraud detection analysis dataset
##The data model is comprised of the following four tables:

beneficiary_data – Stores comprehensive patient information, including demographics and health conditions.

inpatient_data – Captures details of hospital admissions and associated provider information.

outpatient_data – Records information related to outpatient visits and the corresponding healthcare providers.

fraudulent_provider – Serves as a reference to label and identify whether a provider is involved in fraudulent activities.

## Key Highlights

- ⚕️ **Problem Addressed**: Detection of fraudulent claims by healthcare providers, improving the integrity of healthcare systems.

- 🧠 **Machine Learning-Based Detection**: 
  - Advanced machine learning models are developed using **Python** and **Scikit-learn**.
  - These models are trained to recognize patterns and anomalies indicative of fraud.

- 🗃️ **Database Management**:
  - A **PostgreSQL** database is deployed on **AWS EC2** for reliable and efficient data storage and querying.
  - Structured data from multiple sources (inpatient, outpatient, beneficiary details) is ingested and merged for analysis.

- ☁️ **Scalable Storage**:
  - **AWS S3** is used for storing raw and processed data files securely and at scale.

- 📊 **Visualization and Reporting**:
  - **Power BI** dashboards provide interactive, real-time visualizations of prediction results.
  - Stakeholders can monitor trends, review flagged providers, and gain insights into healthcare claim patterns.

- 🧹 **ETL Pipeline**:
  - Custom Python scripts perform extract, transform, and load (ETL) operations on the healthcare datasets.
  - Data is preprocessed, cleaned, and features are engineered to support model training.

- 🔐 **Goal**:
  - Enhance fraud detection capabilities.
  - Support proactive intervention and policy decisions with accurate and explainable results.

## Technologies Used

- **Python** (pandas, scikit-learn, matplotlib, seaborn)
- **PostgreSQL** (hosted on AWS EC2)
- **AWS S3** (cloud storage)
- **Power BI** (dashboard and prediction visualization)
- **Jupyter Notebook** (for model development and analysis)
- **EC2** (for connection)
- **matplotlib** (Data Visualization)
-**scikit-learn** (Machine learning models)




