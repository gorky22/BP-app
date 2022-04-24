# Bachelor's Thesis
SIMPLE RECOMMENDATION SYSTEM

Author: Damian Gorcak
Supervisor: Ing. VLADIMÍR BARTÍK, Ph.D.

## Implementation

App is implemented in mostly in Python3. It uses 3 methods of Colaborative filtering (SVD, ALS, SGD). Backend is implemented with Library Flask, where datasets are stored, and pages for rendering are created. In app.py for each dataset is created instance of class Dataset, which is implemented in file statistic.py. In this class Dataframe is created, statistic about this dataset is created, also hyperparameters are tuning and prediction are made (with statistics). In statistic.py is used for predictions library Surprise for tuning hyperparameters bayes_opt library, and for making statistic and procesing dataset is used Pandas. For Creating interacting graphs in forntend is used Charts.js

## GUI

First we must add file with datasets.

<img width="1440" alt="form" src="https://user-images.githubusercontent.com/57815724/164977842-7b74f62b-761a-414a-aaf7-c78d1aca0a53.png">

Then in tab "statistic" we can see maded statistic. for example: 

<img width="769" alt="Snímka obrazovky 2022-04-24 o 15 05 27" src="https://user-images.githubusercontent.com/57815724/164978535-4d673eb0-76e5-4fea-bacd-fb9d7d1ff99a.png">

<img width="769" alt="Snímka obrazovky 2022-04-24 o 15 05 27" src="https:<img width="811" alt="Snímka obrazovky 2022-04-24 o 15 05 36" src="https://user-images.githubusercontent.com/57815724/164978013-df31560c-e262-4a47-8cb3-543c117ba4fa.png">
//user-images.githubusercontent.com/57815724/164978012-4934c51f-a402-4afe-aa1e-2b4d7a250a53.png">

in section "find hyperparameter" we can find optimal values of Learning rate and number of Steps. Results are stored and can be shown in section result-hyperparams. Result is shown in picture below.

<img width="607" alt="Snímka obrazovky 2022-04-24 o 15 11 44" src="https://user-images.githubusercontent.com/57815724/164978207-8214e127-6e5c-43ab-bb5b-1b7fc0a1f441.png">

Predictions are inputed with toggle buttons in section find predictions. User can add rating for randomly generated 100 items from dataset, and can choose which algorithm use for making prediction (algorithms are called with learning rate 0.005 and number of steps 30) or can just add rating to dataset and in section "train model"  train model with adding custom parameters (learning rate and number of steps). After creating model and making predictions statistics are shown. 

<img width="1293" alt="odp" src="https://user-images.githubusercontent.com/57815724/164978446-2a3b92d4-af48-4da9-8d39-828dad93ec73.png">

<img width="350" alt="ukazka" src="https://user-images.githubusercontent.com/57815724/164978451-132975fd-7fac-46f9-b930-714a5219de45.png">


