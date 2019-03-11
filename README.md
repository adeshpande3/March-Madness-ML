# March-Madness-ML

Applying machine learning to March Madness. Check out my first repo [here](https://github.com/adeshpande3/March-Madness-2017) and my associated [blog post](https://adeshpande3.github.io/adeshpande3.github.io/Applying-Machine-Learning-to-March-Madness). I've tried to make this repository extensible enough so that I can use it from year to year. 

## Overview

In this project, I hope to use machine learning to create a model that can predict the winner of a game between two teams. This way, I can try to predict the winner of the NCAA Basketball Tournament (and hopefully get a perfect bracket LOL). I've separated this project into a couple of different components. Since I like to do this every year, I wanted to keep this code general enough so that it can work from year to year, you'll just have to add new data for the current year. 

* Data: The Data folder contains different CSVs that show team stats, regular season game results, etc. It will contain data that I've scraped, data from Kaggle, and a folder that contains precomputed xTrain and yTrain matrices so that we don't have to keep recomputing the training set. 
* DataPreprocessing.py: Script where we create our training matrices. 
* MarchMadness2018.py: Script where we apply machine learning models to the training set. We can also create our Kaggle submissions here. 

## Requirements and Installation
In order to run these scripts, you'll need the following libraries. 

* [NumPy](https://docs.scipy.org/doc/numpy/user/install.html)
* [Pandas](https://pandas.pydata.org/pandas-docs/stable/install.html)
* [Sklearn](http://scikit-learn.org/stable/install.html)

## What To Do Every March
* Download data files from Kaggle, who will normally have a competition going. They will provide CSV files that show the results from games since 1985, information on conferences, tourney seed history, etc. It's important to download this data every year because Kaggle will add data from the most recently completed season and so you'll have a bit more training data. **Download the files, and replace the ones in [here](https://github.com/adeshpande3/March-Madness-ML/tree/master/Data/KaggleData) with the new versions**
* We also want to get the advanced rating statistics from Basketball Reference. Basically, go to https://www.sports-reference.com/cbb/seasons/2019-ratings.html (and replace 2019 with whatever year you're looking at), choose to get the table as a CSV (available in one of the dropdowns), copy that over to a new text doc in Sublime (or any text editor), save it as a CSV, and then upload it to [this folder](https://github.com/adeshpande3/March-Madness-ML/tree/master/Data/RatingStats).
* We also want to get the regular season statistics from Basketball Reference. Basically, go to https://www.sports-reference.com/cbb/seasons/2019-school-stats.html (and replace 2019 with whatever year you're looking at), choose to get the table as a CSV (available in one of the dropdowns), copy that over to a new text doc in Sublime (or any text editor), save it as a CSV, and then upload it to [this folder](https://github.com/adeshpande3/March-Madness-ML/tree/master/Data/RegSeasonStats).
* Run DataPreprocessing.py in order to get the most up to date training matrices.
* Run MarchMadness.py. 

## What You Can Do
* Try to modify MarchMadness2018.py to include more ML models
* Modify DataPreprocessing.py to create different features to represent each game/team
* Perform data visualizations to see which features are the most important
* Decide what type of additional data preprocessing might be needed

## Getting Started
1. Download and unzip [this entire repository from GitHub](https://github.com/adeshpande3/March-Madness-2018), either interactively, or by entering the following in your Terminal.
    ```bash
    git clone https://github.com/adeshpande3/March-Madness-2018.git
    ```
2. Navigate into the top directory of the repo on your machine
    ```bash
    cd March-Madness-2018
    ```
3. First create your xTrain and yTrain matrices by running 
    ```bash
    python DataPreprocessing.py
    ```
   This may take a while (Still trying to figure out ways to make this faster).
4. Then run your machine learning model  
    ```bash
    python MarchMadness2018.py
    ```
