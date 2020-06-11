# Class-competition

## Group members: Joshua Diedrich, Race Stewart, Grayland Lunn

The report and Kaggle screenshot can be found as Report.pdf and kaggle_submission

The python code shoud run on the OSU flip server if all imports are installed.  These include:

seaborn
matplotlib
numpy
pandas
sklearn

The python file can be run using python3 on the flip server, and it will run an internal validation test with a 80/20 train/test split.
Runtime can be quite long, and we were able to see similar accuracy by changing the split to test_size=0.01 on line 106.
This can be used to just get a quick idea of accuracy, but a larger split is recommended.
