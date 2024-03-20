
from django.db.models import  Count, Avg
from django.shortcuts import render, redirect
from django.db.models import Count
from django.db.models import Q
import datetime
import xlwt
from django.http import HttpResponse


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


def serviceproviderlogin(request):
    if request.method  == "POST":
        admin = request.POST.get('username')
        password = request.POST.get('password')
        if admin == "Admin" and password =="Admin":
            detection_accuracy.objects.all().delete()
            return redirect('View_Remote_Users')

    return render(request,'SProvider/serviceproviderlogin.html')

def View_Cyber_Attack_Type_Ratio(request):
    detection_ratio.objects.all().delete()
    ratio = ""
    kword = 'Reconnaissance'
    print(kword)
    obj = cyberattack_detection_Type.objects.all().filter(Q(Prediction=kword))
    obj1 = cyberattack_detection_Type.objects.all()
    count = obj.count();
    count1 = obj1.count();
    ratio = (count / count1) * 100
    if ratio != 0:
        detection_ratio.objects.create(names=kword, ratio=ratio)

    ratio1 = ""
    kword1 = 'Exploits'
    print(kword1)
    obj1 = cyberattack_detection_Type.objects.all().filter(Q(Prediction=kword1))
    obj11 = cyberattack_detection_Type.objects.all()
    count1 = obj1.count();
    count11 = obj11.count();
    ratio1 = (count1 / count11) * 100
    if ratio1 != 0:
        detection_ratio.objects.create(names=kword1, ratio=ratio1)

    ratio12 = ""
    kword12 = 'DoS'
    print(kword12)
    obj12 = cyberattack_detection_Type.objects.all().filter(Q(Prediction=kword12))
    obj112 = cyberattack_detection_Type.objects.all()
    count12 = obj12.count();
    count112 = obj112.count();
    ratio12 = (count12 / count112) * 100
    if ratio12 != 0:
        detection_ratio.objects.create(names=kword12, ratio=ratio12)

    ratio123 = ""
    kword123 = 'Generic'
    print(kword123)
    obj123 = cyberattack_detection_Type.objects.all().filter(Q(Prediction=kword123))
    obj1123 = cyberattack_detection_Type.objects.all()
    count123 = obj123.count();
    count1123 = obj1123.count();
    ratio123 = (count123 / count1123) * 100
    if ratio123 != 0:
        detection_ratio.objects.create(names=kword123, ratio=ratio123)

    ratio1234 = ""
    kword1234 = 'Fuzzers'
    print(kword1234)
    obj1234 = cyberattack_detection_Type.objects.all().filter(Q(Prediction=kword1234))
    obj11234 = cyberattack_detection_Type.objects.all()
    count1234 = obj1234.count();
    count11234 = obj11234.count();
    ratio1234 = (count1234 / count11234) * 100
    if ratio1234 != 0:
        detection_ratio.objects.create(names=kword1234, ratio=ratio1234)

    ratioA = ""
    kwordA = 'Worms'
    print(kwordA)
    objA = cyberattack_detection_Type.objects.all().filter(Q(Prediction=kwordA))
    objB = cyberattack_detection_Type.objects.all()
    countA = objA.count();
    countB = objB.count();
    ratioA= (countA / countB) * 100
    if ratioA != 0:
        detection_ratio.objects.create(names=kwordA,ratio=ratioA)

    ratioA1 = ""
    kwordA1 = 'Shellcode'
    print(kwordA1)
    objA1 = cyberattack_detection_Type.objects.all().filter(Q(Prediction=kwordA1))
    objB1 = cyberattack_detection_Type.objects.all()
    countA1 = objA1.count();
    countB1 = objB1.count();
    ratioA1 = (countA1 / countB1) * 100
    if ratioA1 != 0:
        detection_ratio.objects.create(names=kwordA1, ratio=ratioA1)

    ratioA2 = ""
    kwordA2 = 'Backdoors'
    print(kwordA2)
    objA2 = cyberattack_detection_Type.objects.all().filter(Q(Prediction=kwordA2))
    objB2 = cyberattack_detection_Type.objects.all()
    countA2 = objA2.count();
    countB2 = objB2.count();
    ratioA2 = (countA2 / countB2) * 100
    if ratioA2 != 0:
        detection_ratio.objects.create(names=kwordA2, ratio=ratioA2)

    obj = detection_ratio.objects.all()
    return render(request, 'SProvider/View_Cyber_Attack_Type_Ratio.html', {'objs':obj})

def View_Remote_Users(request):
    obj=ClientRegister_Model.objects.all()
    return render(request,'SProvider/View_Remote_Users.html',{'objects':obj})

def ViewTrendings(request):
    topic = cyberattack_detection_Type.objects.values('topics').annotate(dcount=Count('topics')).order_by('-dcount')
    return  render(request,'SProvider/ViewTrendings.html',{'objects':topic})

def charts(request,chart_type):
    chart1 = detection_ratio.objects.values('names').annotate(dcount=Avg('ratio'))
    return render(request,"SProvider/charts.html", {'form':chart1, 'chart_type':chart_type})

def charts1(request,chart_type):
    chart1 = detection_accuracy.objects.values('names').annotate(dcount=Avg('ratio'))
    return render(request,"SProvider/charts1.html", {'form':chart1, 'chart_type':chart_type})

def View_Prediction_Of_Cyber_Attack_Type(request):
    obj =cyberattack_detection_Type.objects.all()
    return render(request, 'SProvider/View_Prediction_Of_Cyber_Attack_Type.html', {'list_objects': obj})

def likeschart(request,like_chart):
    charts =detection_accuracy.objects.values('names').annotate(dcount=Avg('ratio'))
    return render(request,"SProvider/likeschart.html", {'form':charts, 'like_chart':like_chart})


def Download_Trained_DataSets(request):

    response = HttpResponse(content_type='application/ms-excel')
    # decide file name
    response['Content-Disposition'] = 'attachment; filename="TrainedData.xls"'
    # creating workbook
    wb = xlwt.Workbook(encoding='utf-8')
    # adding sheet
    ws = wb.add_sheet("sheet1")
    # Sheet header, first row
    row_num = 0
    font_style = xlwt.XFStyle()
    # headers are bold
    font_style.font.bold = True
    # writer = csv.writer(response)
    obj = cyberattack_detection_Type.objects.all()
    data = obj  # dummy method to fetch data.
    for my_row in data:
        row_num = row_num + 1
        ws.write(row_num, 0, my_row.Source_Ip, font_style)
        ws.write(row_num, 1, my_row.Destination_Ip, font_style)
        ws.write(row_num, 2, my_row.Attack_Details, font_style)
        ws.write(row_num, 3, my_row.Prediction, font_style)

    wb.save(response)
    return response

def train_model(request):
    detection_accuracy.objects.all().delete()

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
    detection_accuracy.objects.create(names="Naive Bayes", ratio=naivebayes)

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
    detection_accuracy.objects.create(names="SVM", ratio=svm_acc)

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
    detection_accuracy.objects.create(names="Logistic Regression", ratio=accuracy_score(y_test, y_pred) * 100)

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
    detection_accuracy.objects.create(names="Decision Tree Classifier", ratio=accuracy_score(y_test, dtcpredict) * 100)

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
    detection_accuracy.objects.create(names="KNeighborsClassifier", ratio=accuracy_score(y_test, knpredict) * 100)

    Labled_Data = 'Labled_Data.csv'
    df.to_csv(Labled_Data, index=False)
    df.to_markdown

    obj = detection_accuracy.objects.all()
    return render(request,'SProvider/train_model.html', {'objs': obj})