# User-Authentication-using-Keystroke-Data
Authenticate the user by training and testing One Class Support Vector Machine classifier by keystroke dynamics of the neutral, happy and sad mood data.   
Keystroke dynamics is one of the most efficient and inexpensive behavioral biometrics that can be used to authenticate a user. So SVM, along with keystroke dynamics can be utilized as the classification engine of users with high efficacy due to its high recognition rate and efficient processing.  


# Requirements to run the python code:

1. Python 3

2. Python modules needed: scikit-learn
                          pandas
                          numpy

3. The dataset file keystroke.csv must be present in the same folder as svm.py

4. The python code is in file named svm.py

5. Open terminal. Change dir to where svm.py and csv files are located.

6. Run command 'python svm.py'.

7. The program takes operates on the data file keystroke.csv, outputs validation error rates
   and also produces a file named keymean.csv that has all the NaN data replaced.

