# March-Madness-ML

Applying machine learning to March Madness. Check out my first repo [here](https://github.com/adeshpande3/March-Madness-2017) and my associated [blog post](https://adeshpande3.github.io/adeshpande3.github.io/Applying-Machine-Learning-to-March-Madness). I've tried to make this repository extensible enough so that I can use it from year to year. 

## Overview

In this project, I hope to use machine learning to create a model that can predict the winner of a game between two teams. This way, I can try to predict the winner of the NCAA Basketball Tournament (and hopefully get a perfect bracket LOL). I've separated this project into a couple of different components. Since I like to do this every year, I wanted to keep this code general enough so that it can work from year to year, you'll just have to add new data for the current year. 

* Data: The Data folder contains different CSVs that show team stats, regular season game results, etc. It will contain data that I've scraped, data from Kaggle, and a folder that contains precomputed xTrain and yTrain matrices so that we don't have to keep recomputing the training set. 
* DataPreprocessing.py: Script where we create our training matrices. 
* MarchMadness.py: Script where we apply machine learning models to the training set. We can also create our Kaggle submissions here. 

## Requirements and Installation

* python 3
* [pipenv](https://pipenv.readthedocs.io/en/latest/) for managing virtualenv and pip package dependencies.

## What To Do Every March
* Download data files from [Kaggle](https://www.kaggle.com/c/mens-machine-learning-competition-2019), who will normally have a competition going (look for the competition for the current year). They will provide CSV files that show the results from games since 1985, information on conferences, tourney seed history, etc. It's important to download this data every year because Kaggle will add data from the most recently completed season and so you'll have a bit more training data. **Download the files, and replace the ones in [here](https://github.com/adeshpande3/March-Madness-ML/tree/master/Data/KaggleData) with the new versions**
* We also want to get the advanced rating statistics from Basketball Reference. Basically, go to https://www.sports-reference.com/cbb/seasons/2019-ratings.html, replace 2019 with whatever year you're looking at, choose to get the table as a CSV (available in one of the dropdowns), disregard the first line, start with the line that begins with "Rk,School..", copy that over to a new text doc in Sublime (or any text editor), save it as a CSV, and then upload it to [this folder](https://github.com/adeshpande3/March-Madness-ML/tree/master/Data/RatingStats).
* We also want to get the regular season statistics from Basketball Reference. Basically, go to https://www.sports-reference.com/cbb/seasons/2019-school-stats.html, replace 2019 with whatever year you're looking at, choose to get the table as a CSV (available in one of the dropdowns), disregard the first line, start with the line that begins with "Rk,School..", copy that over to a new text doc in Sublime (or any text editor), save it as a CSV, and then upload it to [this folder](https://github.com/adeshpande3/March-Madness-ML/tree/master/Data/RegSeasonStats).
    * For both of the above steps, make sure that the column names are the same from year to year! In 2019, Basketball Reference made some small changes to the column names (X3P to 3PA for example)
* Run DataPreprocessing.py in order to get the most up to date training matrices.
* Run MarchMadness.py. 

## What You Can Do
* Try to modify MarchMadness.py to include more ML models
* Modify DataPreprocessing.py to create different features to represent each game/team
* Perform data visualizations to see which features are the most important
* Decide what type of additional data preprocessing might be needed

## Getting Started
1. Download and unzip [this entire repository from GitHub](https://github.com/adeshpande3/March-Madness-2018), either interactively, or by entering the following in your Terminal.
    ```bash
    git clone https://github.com/adeshpande3/March-Madness-ML.git
    ```
2. Navigate into the top directory of the repo on your machine
    ```bash
    cd March-Madness-ML
    ```
3. Create a virtualenv and install the package dependencies. If you don't have `pipenv`, you can follow instructions [here](https://pipenv.pypa.io/en/latest/install/) for how to install.
    ```bash
    pipenv install
    ```
4. First create your xTrain and yTrain matrices by running 
    ```bash
    pipenv run python DataPreprocessing.py
    ```
   This may take a while (Still trying to figure out ways to make this faster).
5. Then run your machine learning model  
    ```bash
    pipenv run python MarchMadness.py
    ```
## Troubleshooting
* If you're using Python 2, then everything should be the same except you don't have to create a pipenv, but you would have to install the following libraries on your own: numpy, pandas, sklearn. Other optional libraries are keras, tensorflow, and xgboost. 
* If you are using the pipenv with Python 3.7 approach and you want to use Tensorflow, you might run into issues with versioning like [this one](https://github.com/adeshpande3/March-Madness-ML/issues/13). The tl;dr is to use Python 3.6 instead of 3.7.
* If you are getting errors with any Tensorflow, Keras, or Xgboost installation, keep in mind that those aren't completely necessary for being able to run MarchMadness.py. They are just helpful for if you want to create neural network models (Tensorflow/Keras) or if you want to run Gradient Boosted models (Xgboost). If you are getting errors and you don't really want to use those models, you can go ahead and remove those [import lines](https://github.com/adeshpande3/March-Madness-ML/blob/master/MarchMadness.py#L10-L42). 
