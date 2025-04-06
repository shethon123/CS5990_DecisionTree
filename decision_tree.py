# -------------------------------------------------------------------------
# AUTHOR: Sheldin Lau
# FILENAME: decision_tree.py
# SPECIFICATION: creating a decision tree to test how more training data improves accuracy
# FOR: CS 5990 (Advanced Data Mining) - Assignment #3
# TIME SPENT: 2 Hours
# -----------------------------------------------------------*/

#IMPORTANT NOTE: YOU HAVE TO WORK WITH THE PYTHON LIBRARIES numpy AND pandas to complete this code.

#importing some Python libraries
from sklearn import tree
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

dataSets = ['cheat_training_1.csv', 'cheat_training_2.csv', 'cheat_training_3.csv']

for ds in dataSets:

    X = []
    Y = []

    df = pd.read_csv(ds, sep=',', header=0)   #reading a dataset eliminating the header (Pandas library)
    data_training = np.array(df.values)[:,1:] #creating a training matrix without the id (NumPy library)

    #transform the original training features to numbers and add them to the 5D array X. For instance, Refund = 1, Single = 1, Divorced = 0, Married = 0,
    #Taxable Income = 125, so X = [[1, 1, 0, 0, 125], [2, 0, 1, 0, 100], ...]]. The feature Marital Status must be one-hot-encoded and Taxable Income must
    #be converted to a float.
    X = []
    for i in data_training:
        tmp = []
        for j in i:
            if j == 'Yes':
                tmp.append(1)
            elif j == 'No':
                tmp.append(0)
            elif j == 'Divorced':
                tmp.append(1)
                tmp.append(0)
                tmp.append(0)
            elif j == 'Married':
                tmp.append(0)
                tmp.append(1)
                tmp.append(0)
            elif j == 'Single':
                tmp.append(0)
                tmp.append(0)
                tmp.append(1)
            else:
                removeK = j.strip('k')
                tmp.append(int(removeK))
                break
        X.append(tmp)

    #transform the original training classes to numbers and add them to the vector Y. For instance Yes = 1, No = 2, so Y = [1, 1, 2, 2, ...]
    #--> add your Python code here
    Y = []
    for i in df['Cheat']:
        if i == 'Yes':
            Y.append(1)
        else:
            Y.append(0)

    #loop your training and test tasks 10 times here
    accuracy = []
    for i in range (10):
       #fitting the decision tree to the data by using Gini index and no max_depth
       clf = tree.DecisionTreeClassifier(criterion = 'gini', max_depth=None)
       clf = clf.fit(X, Y)

       #plotting the decision tree
       tree.plot_tree(clf, feature_names=['Refund', 'Single', 'Divorced', 'Married', 'Taxable Income'], class_names=['Yes','No'], filled=True, rounded=True)
       # plt.show()

       #read the test data and add this data to data_test NumPy
       #--> add your Python code here
       df = pd.read_csv('cheat_test.csv', sep=',', header=0)  # reading a dataset eliminating the header (Pandas library)
       data_test = np.array(df.values)[:, 1:]
       counter  = 0

       for data in data_test:
           #transform the features of the test instances to numbers following the same strategy done during training, and then use the decision tree to make the class prediction. For instance:
           #class_predicted = clf.predict([[1, 0, 1, 0, 115]])[0], where [0] is used to get an integer as the predicted class label so that you can compare it with the true label
           #--> add your Python code here
           tmp = []
           for j in data:
               if j == 'Yes':
                   tmp.append(1)
               elif j == 'No':
                   tmp.append(0)
               elif j == 'Divorced':
                   tmp.append(1)
                   tmp.append(0)
                   tmp.append(0)
               elif j == 'Married':
                   tmp.append(0)
                   tmp.append(1)
                   tmp.append(0)
               elif j == 'Single':
                   tmp.append(0)
                   tmp.append(0)
                   tmp.append(1)
               else:
                   removeK = j.strip('k')
                   tmp.append(int(removeK))
                   break
           class_predicted = clf.predict([tmp])[0]
           #compare the prediction with the true label (located at data[3]) of the test instance to start calculating the model accuracy.
           #--> add your Python code here
           if data[3] == 'Yes' and class_predicted == 1:
               counter += 1
           elif data[3] == 'No' and class_predicted == 0:
               counter += 1
       accuracy.append(counter/len(data_test))
       #find the average accuracy of this model during the 10 runs (training and test set)
       #--> add your Python code here
    averageAccuracy = sum(accuracy)/10
    #print the accuracy of this model during the 10 runs (training and test set).
    #your output should be something like that: final accuracy when training on cheat_training_1.csv: 0.2
    #--> add your Python code here
    print('final accuracy when training on cheat_training_1.csv: ', averageAccuracy)



