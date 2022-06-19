# Disclaimer
All of my repositories (including this one) are NOT actively maintained. I have disabled issues, so if you decide to download or use any of my projects, you will troubleshoot any issues yourself. You can safely assume that most of my repo's are in a "proof-of-concept" stage; some of them are completed and working, others are just templates that need more work. The repo's I choose to post on GitHub are meant for review, storage and job applications. Thank you.

# Project-1-Closing-Price-Stock-Regressor-Version-1.0
Predicting a stock's daily closing price using automated multiple linear regression. The Jupyter Notebook in Python 3.6 (Windows 10) has already been run and the outputs are uploaded in this repository. If you wish to re-run it, make sure you Clear All Ouputs and Restart the kernel. You will have new output plot images and .csv/.txt summaries generated for your run.

This is my first repository on GitHub. These projects are meant to be a display of my current Python and machine learning programming knowledge to date (August 2019). This first project has been well commented and I'm hoping that someone new to the field will find it useful in getting started.

# Introduction

This Jupyter notebook attempts (keyword, attempts) to predict the daily closing price of an inputted stock ticker, given its historical data, some added features through feature engineering, and the current day's opening price. I am going to do my best to explain each detail so that anyone (newbie or professional) can follow along. This model as it stands right now is not very good at predicting the closing price, but that isn't really my intent with this project, but more just to display what I've learned so far, and for others to get started as well if they're interested in a skeleton project to get started. There's always room for improvement but I'm hoping that this will help get someone started on their way if they are looking to get into the field of machine learning.

Day traders typically start their mornings by creating a 'watchlist' of stocks before the market's opening bell. The theory behind this project is that, if you know (roughly) where a stock is about to open today pre-market, and have some of its previous days historical data, you might be able to train a regression model to gauge a better idea of where the stock will close that day, giving actionable insight into whether to buy the stock or not.

This project is only meant to act as a demonstration of some of my Python programming and machine learning knowledge to date (August, 2019 as of this writing). Everything I've compiled in this project I have pieced together from various sources and it's all things that I have learned from my own research and self-teaching. I have many years of technical analysis knowledge of the stock and Forex markets, but I do not have a degree in Computer Science or any formal education in the field (which may or may not become evident to whomever is reading this). I welcome advice and criticisms to make things more efficient and better so that I can develop as a programmer. If you see anything I've missed the point on, please let me know. I have spent a fair amount of time commenting each section of code to make it easy for a newbie coder to read along and understand what's happening. Experienced coders will be able to just look at each code section and see what's going on, so skim or read, the choice is yours. This should also help to isolate a section in case someone runs into an issue, they will be able to point to where it happened.



    DISCLAIMER: This project is merely a compilation of various Python and machine learning skills that I have picked up 
    over the past few months and years, and acting as a test for me to see if I'm able to CLEARLY explain the various 
    details and tools used throughout the project so that maybe one day I can show a potential employer my skillset and 
    secure a job in this field. This tool is not actually meant to be used for stock advisement, it is for academic
    and future development purposes only.

    You've been warned. Now enjoy!
    
    
    
# The Environment and Installing the Dependencies

First thing's first, if you want to run this project, you need to have the appropriate dependencies installed. Everything listed below are the versions of each dependency as of this writing. I'll explain how to install them all at the bottom of the list. Now I can't speak for other versions, but you need to be running ... Python 3.6 on Windows 10 ..., as that's what I'm running on my machine. Here are the dependencies you'll need:

1. DateTime==4.3
2. pandas==0.23.0
3. pandas-datareader==0.7.0
4. seaborn==0.8.1
5. matplotlib==2.1.2
6. numpy==1.14.5
7. bs4==0.0.1
8. requests==2.22.0
9. feature-selector==1.0.0
10. featuretools==0.9.1
11. dask==2.2.0
12. dask-ml==1.0.0
13. sklearn==0.0
14. TPOT==0.10.2
15. ta-lib==0.4.18

Again, this is all in Python 3.6 and Windows 10. All of the other modules used in this project (like "os" and "warnings" should be included in your Python package already. If you run into something I've missed, let me know, but this should be a complete list. There are two ways to install all of these packages:

    1. Install them all at once using the Pip installer by typing the following command into your command prompt:

    pip install DateTime==4.3 pandas==0.23.0 pandas-datareader==0.7.0 seaborn==0.8.1 matplotlib==2.1.2 numpy==1.14.5 bs4==0.0.1 requests==2.22.0 feature-selector==1.0.0 featuretools==0.9.1 dask==2.2.0 dask-ml==1.0.0 sklearn TPOT==0.10.2 TA-Lib

Or

    pip3 install DateTime==4.3 pandas==0.23.0 pandas-datareader==0.7.0 seaborn==0.8.1 matplotlib==2.1.2 numpy==1.14.5 bs4==0.0.1 requests==2.22.0 feature-selector==1.0.0 featuretools==0.9.1 dask==2.2.0 dask-ml==1.0.0 sklearn TPOT==0.10.2 TA-Lib

(Depending on how you have installed your Pip installer. Notice the 3 after pip)

    2. Install them all from the provided "requirements.txt" file by typing in the following command into your command prompt:

    pip install -r requirements.txt

Or

    pip3 install -r requirements.txt

(One of those two ways should create the environment needed to run this project. Otherwise each module will need to be installed manually as needed).


# References and Resources Before We Get Started

Before I forget and start going down the code wormhole, I wanted to take some time and reference some various websites, other's projects and the tools that I'm using in the following code. I want to thank everyone involved in my learning process and I urge you to check them out:

1. The motivation and skeleton for my project - https://gogul09.github.io/software/regression-example-boston-housing-prices (That article taught me more in one reading than any other)

2. TPOT (automated machine learning) - https://epistasislab.github.io/tpot/
3. FeatureTools (feature engineering) - https://www.featuretools.com/
4. Feature-Selector (take a guess :D) - https://github.com/WillKoehrsen/feature-selector
5. Dask (parallel computing) - https://docs.dask.org/en/latest/
6. TA-Lib (technical indicators) - https://mrjbq7.github.io/ta-lib/index.html

7. Medium (Great resource for learning. Sign up for emails based on what you read!) - https://medium.com/topic/artificial-intelligence
8. Towards Data Science (Another awesome resource) - https://towardsdatascience.com/
9. Machine Learning Mastery (see above) - https://machinelearningmastery.com/
10. Stack Overflow (duh!) - https://stackoverflow.com/
11. Upwork (in a real pinch, pay someone to teach you something) - www.upwork.com
12. Udemy (I received great value from this course) - https://www.udemy.com/the-data-science-course-complete-data-science-bootcamp/

# Enjoy!
