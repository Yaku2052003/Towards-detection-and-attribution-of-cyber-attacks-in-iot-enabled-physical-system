from django.db.models import Count
from django.db.models import Q
from django.shortcuts import render, redirect, get_object_or_404
import datetime
import openpyxl

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from sklearn.ensemble import VotingClassifier
from sklearn.tree import DecisionTreeClassifier
import warnings
warnings.filterwarnings("ignore")
plt.style.use('ggplot')
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
# Create your views here.
from Remote_User.models import ClientRegister_Model,cyberattack_detection_Type,detection_ratio,detection_accuracy

def login(request):


    if request.method == "POST" and 'submit1' in request.POST:

        username = request.POST.get('username')
        password = request.POST.get('password')
        try:
            enter = ClientRegister_Model.objects.get(username=username,password=password)
            request.session["userid"] = enter.id

            return redirect('ViewYourProfile')
        except:
            pass

    return render(request,'RUser/login.html')

def Add_DataSet_Details(request):

    return render(request, 'RUser/Add_DataSet_Details.html', {"excel_data": ''})


def Register1(request):

    if request.method == "POST":
        username = request.POST.get('username')
        email = request.POST.get('email')
        password = request.POST.get('password')
        phoneno = request.POST.get('phoneno')
        country = request.POST.get('country')
        state = request.POST.get('state')
        city = request.POST.get('city')
        ClientRegister_Model.objects.create(username=username, email=email, password=password, phoneno=phoneno,
                                            country=country, state=state, city=city)

        return render(request, 'RUser/Register1.html')
    else:
        return render(request,'RUser/Register1.html')

def ViewYourProfile(request):
    userid = request.session['userid']
    obj = ClientRegister_Model.objects.get(id= userid)
    return render(request,'RUser/ViewYourProfile.html',{'object':obj})


def Predict_Cyber_Attack_Type(request):
    if request.method == "POST":
        review = request.POST.get('keyword')
        if request.method == "POST":
            sip = request.POST.get('sip')
            dip = request.POST.get('dip')
            attack = request.POST.get('keyword')

        df = pd.read_csv('IOT_Cyber_Attacks.csv')
        df
        df.columns
        df.isnull().sum()
        df.rename(columns={'Attack_category': 'Acat', 'Attack_Name': 'Atype'}, inplace=True)

        def apply_find(Acat):
            if (Acat == "Reconnaissance"):
                return 0  # Reconnaissance
            elif (Acat == "Exploits"):
                return 1  # Exploits
            elif (Acat == "DoS"):
                return 2  # DoS
            elif (Acat == "Generic"):
                return 3  # Generic
            elif (Acat == "Fuzzers"):
                return 4  # Fuzzers
            elif (Acat == "Worms"):
                return 5  # Worms
            elif (Acat == "Shellcode"):
                return 6  # Shellcode
            elif (Acat == "Backdoors"):
                return 7  # Backdoors

        df['Label'] = df['Acat'].apply(apply_find)
        df.drop(['Acat'], axis=1, inplace=True)
        Label = df['Label'].value_counts()
        # df.drop(['id','timestamp','url','replies','retweets','quotes'], axis=1, inplace=True)

        cv = CountVectorizer()
        X = df['Atype']
        y = df['Label']

        print("Atype")
        print(X)
        print("Label")
        print(y)

        X = cv.fit_transform(X)

        models = []
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
        X_train.shape, X_test.shape, y_train.shape

        print("Naive Bayes")

        from sklearn.naive_bayes import MultinomialNB
        NB = MultinomialNB()
        NB.fit(X_train, y_train)
        predict_nb = NB.predict(X_test)
        naivebayes = accuracy_score(y_test, predict_nb) * 100
        print(naivebayes)
        print(confusion_matrix(y_test, predict_nb))
        print(classification_report(y_test, predict_nb))
        models.append(('naive_bayes', NB))

        # SVM Model
        print("SVM")
        from sklearn import svm
        lin_clf = svm.LinearSVC()
        lin_clf.fit(X_train, y_train)
        predict_svm = lin_clf.predict(X_test)
        svm_acc = accuracy_score(y_test, predict_svm) * 100
        print(svm_acc)
        print("CLASSIFICATION REPORT")
        print(classification_report(y_test, predict_svm))
        print("CONFUSION MATRIX")
        print(confusion_matrix(y_test, predict_svm))
        models.append(('svm', lin_clf))

        print("Logistic Regression")

        from sklearn.linear_model import LogisticRegression
        reg = LogisticRegression(random_state=0, solver='lbfgs').fit(X_train, y_train)
        y_pred = reg.predict(X_test)
        print("ACCURACY")
        print(accuracy_score(y_test, y_pred) * 100)
        print("CLASSIFICATION REPORT")
        print(classification_report(y_test, y_pred))
        print("CONFUSION MATRIX")
        print(confusion_matrix(y_test, y_pred))
        models.append(('logistic', reg))

        print("Decision Tree Classifier")
        dtc = DecisionTreeClassifier()
        dtc.fit(X_train, y_train)
        dtcpredict = dtc.predict(X_test)
        print("ACCURACY")
        print(accuracy_score(y_test, dtcpredict) * 100)
        print("CLASSIFICATION REPORT")
        print(classification_report(y_test, dtcpredict))
        print("CONFUSION MATRIX")
        print(confusion_matrix(y_test, dtcpredict))
        models.append(('DecisionTreeClassifier', dtc))

        print("KNeighborsClassifier")
        from sklearn.neighbors import KNeighborsClassifier
        kn = KNeighborsClassifier()
        kn.fit(X_train, y_train)
        knpredict = kn.predict(X_test)
        print("ACCURACY")
        print(accuracy_score(y_test, knpredict) * 100)
        print("CLASSIFICATION REPORT")
        print(classification_report(y_test, knpredict))
        print("CONFUSION MATRIX")
        print(confusion_matrix(y_test, knpredict))
        models.append(('KNeighborsClassifier', kn))

        classifier = VotingClassifier(models)
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)

        tweet_data = [attack]
        vector1 = cv.transform(tweet_data).toarray()
        predict_text = classifier.predict(vector1)

        pred = str(predict_text).replace("[", "")
        pred1 = pred.replace("]", "")

        prediction = int(pred1)

        if (prediction == 0):
            val = "Reconnaissance"
        elif (prediction == 1):
            val = "Exploits"
        elif (prediction == 2):
            val = "DoS"
        elif (prediction == 3):
            val = "Generic"
        elif (prediction == 4):
            val = "Fuzzers"
        elif (prediction == 5):
            val = "Worms"
        elif (prediction == 6):
            val = "Shellcode"
        elif (prediction == 7):
            val = "Backdoors"

        print(val)
        print(pred1)

        cyberattack_detection_Type.objects.create(Source_Ip=sip,Destination_Ip=dip,Attack_Details=attack,Prediction=val)

        return render(request, 'RUser/Predict_Cyber_Attack_Type.html',{'objs': val})
    return render(request, 'RUser/Predict_Cyber_Attack_Type.html')



