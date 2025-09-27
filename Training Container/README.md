# Training Container

**Purpose:** Used to train ML models for the personal blood sugar data of the user. Meant to be run as a container inside of a VM or batch job with S3 bucket access.

**Runtime / Deploy Target:** AWS SageMaker (Tensorflow 19 base image for stable version)  
**Entry Point:** `app`  
**Dependencies:** Data to be used must be present inside of a csv file in a so-called "folder" in AWS S3 created for the user.