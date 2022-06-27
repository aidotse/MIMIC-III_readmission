# Predicting 30-day ICU readmissions from the MIMIC-III database


This project was developed as the Capstone project for the Udacity Machine Learning Engineer Nanodegree. Full description and discussion can be found in [Report.pdf](Report.pdf).


## Project Overview

This project describes the development of a model for predicting whether a patient discharged from an intensive care unit (ICU) is likely to be readmitted within 30 days. To do this, I used the MIMIC-III database which contains details records from ~60,000 ICU admissions for ~40,000 patients over a period of 10 years. Using these records, patient records for those readmitted within 30 days of discharge were isolated and used to train a generalized classifier to predict readmission. The project also includes parameter optimization and in depth performance analysis for the classifier.

### Data
This project uses the MIMIC-III database, accessible via:
https://mimic.physionet.org/

To properly run the project from the included files, a local postgres SQL server must be installed and the MIMIC-III database must be set up as described in https://github.com/MIT-LCP/mimic-code/tree/master/buildmimic.

An SQL materialized view was extracted from the database as defined in all_data.sql.

### python libraries used:

- numpy - numerical operation
- pandas - dataframe handling
- os - general operating system operations
- psycopg2 - Used to access a locally installed postgresql server and
perform sql queries.
- xgboost - eXtreme Gradient Boosted trees. Classifier implementation.
- scikit-learn - Used for hyperparameter optimization and performance
metrics evaluation.
- scipy - interp function used during plotting of the ROC curve.
- matplotlib - visualization
- seaborn - visualization

### Setup

1. Get access to the MIMIC database here: https://mimic.mit.edu/docs/gettingstarted/. Download the files for mimic-iii and place them on the server.

2. Check if postgresql is installed by running ```psql --version```. These instuctions where tested using version 12.11, but it is possible that other versions would work as well. If it's not installed, install by following the instructions here: https://www.postgresql.org/download/linux/ubuntu/.

3. Follow the instructions here to set up a postgresql database with the mimic data: https://github.com/MIT-LCP/mimic-code/tree/main/mimic-iii/buildmimic/postgres

4. Run the file all_data.sql by running

   ```\i /path_to_file```

5. Rename this repository to lower case letter. Build and run the docker container:

   ```cd mimic-iii_readmission```


   ```docker build -t mimic-iii_readmission --build-arg USER_ID=$(id -u) --build-arg GROUP_ID=$(id -g) -f Dockerfile .```


   ```docker run -d --rm -it --volume $(pwd):/workspace --gpus all --network="host" --name mimic-iii_readmission mimic-iii_readmission```


   ```docker exec -it mimic-iii_readmission bash```

6. Run scripts 1-4.


### Main results:
Raw numerical data scatter matrix. A: Scatter matrix of all numerical features. B: zoomed-in view on the first 6 features. Each square shows the scatter plot corresponding to the two features defined by the row and column. The diagonal shows the kernel density estimation plots.
![features before preprocessing](figures/scatter_comb_pre.png)

Preprocessed numerical data scatter matrix. A: Scatter matrix of all numerical features. B: zoomed-in view on the first 6 features. Each square shows the scatter plot corresponding to the two features defined by the row and column. The diagonal shows the kernel density estimation plots.
![features after preprocessing](figures/scatter_comb_post.png)

Receiver Operating Characteristic curve for the optimized model, using 5-fold cross validation.
![ROC curve](figures/ROC.png)

Important features. (A) shows the features sorted by “weight”. (B) shows kde plots for the top 9 most important features, separated by label.
![Feature importance plot](figures/features.png)
