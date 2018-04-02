# -*- coding: utf-8 -*-
"""
Created on Thu Mar 15 13:35:05 2018

@author: Mridula
"""

from sklearn import tree
import numpy as np
import pandas as pd
import graphviz 
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import precision_recall_fscore_support
from plotly.offline import download_plotlyjs, init_notebook_mode,  plot
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
#from sklearn.cross_validation import KFold
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import plotly.plotly as py
import plotly.graph_objs as go
import matplotlib.axes
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import matplotlib as mpl
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.cross_validation import train_test_split
from graphviz import Digraph
from sklearn.metrics import mean_squared_error
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
from plotly.offline import download_plotlyjs, init_notebook_mode,  plot # used for plotting graphs
from plotly.graph_objs import *   # used for plotting graphs
from sklearn.metrics import accuracy_score # used for accuracy score calculation
from sklearn.metrics import recall_score # used for macro average recall calculation
from sklearn.metrics import precision_score # used for macro average precision calculation

Data1_wpbc = pd.read_csv('wpbc.csv')
Data1_wpbc_new = Data1_wpbc.drop(['ID number', 'Lymph node status'], axis = 1) 


#train ,test = train_test_split(Data1_wpbc_new,test_size=0.33, stratify=Data1_wpbc_new['Outcome'])
_data = Data1_wpbc_new.values[:,2:32]
_class = Data1_wpbc_new.values[:,0] 

X = pd.DataFrame(data=_data, columns = ['radius1',	'texture1',	'perimeter1',	'area1',	'smoothness1',	'compactness1',	'concavity1',	'concave points 1',	'symmetry1',	'fractal dimension1',	'radius2',	'texture2',	'perimeter2',	'area2',	'smoothness2',	'compactness2',	'concavity2',	'concave points 2',	'symmetry2',	'fractal dimension2',	'radius3',	'texture3',	'perimeter3',	'area3',	'smoothness3',	'compactness3',	'concavity3',	'concave points 3',	'symmetry3',	'fractal dimension3'])
Y = pd.DataFrame(data=_class, columns = ['Outcome'])

centroids=[] 
SSE=[] 
silhouette_coeficient=[]


def indv_SSE(X, label, center):
    SSE0 = SSE1 = SSE2 = SSE3 = 0
    for i in range(len(label)):
        if(label[i]==0):
            for j in range(X.shape[1]):
                SSE0 += (float(X.iloc[i,j]) - center[0][j])**2
        elif(label[i]==1):
            for j in range(X.shape[1]):
                SSE1 += (float(X.iloc[i,j]) - center[1][j])**2
        elif(label[i]==2):
            for j in range(X.shape[1]):
                SSE2 += (float(X.iloc[i,j]) - center[2][j])**2
        elif(label[i]==3):
            for j in range(X.shape[1]):
                SSE3 += (float(X.iloc[i,j]) - center[3][j])**2
                
    return(SSE0, SSE1, SSE2, SSE3) 

for h in range(3):     
                
    kmeans=KMeans(n_clusters=4, n_init=h+3).fit(X)
    print("For run %d, Cluster centers are:" %h )
    print(kmeans.cluster_centers_)
    print("For run %d, Sum of Squared Error is:" %h )
    print(kmeans.inertia_)
    print("For run %d, Labels of data are:" %h )
    print(kmeans.labels_)   
    print("For run %d, Individual SSEs are:" %h )    
    SSE = indv_SSE(X, kmeans.labels_, kmeans.cluster_centers_ )
    print(SSE)
    print("For run %d, Average Silhoutte coeffient is:" %h)     
    sil_coe=silhouette_score(X, kmeans.labels_, metric='euclidean')     
    print(sil_coe)
    silcoe=silhouette_samples(X,kmeans  .labels_,metric='euclidean')
    #############Silhouette Coefficient for K-means#####################
    coe0=[] 
    coe1=[] 
    coe2=[] 
    coe3=[]
    datapoint0=[] 
    datapoint1=[] 
    datapoint2=[]
    datapoint3=[]
    for i in range(len(kmeans.labels_)): 
        if kmeans.labels_[i]==0: 
             coe0.append(silcoe[i])
             datapoint0.append(i)
        elif kmeans.labels_[i]==1: 
             coe1.append(silcoe[i])
             datapoint1.append(i)
        elif kmeans.labels_[i]==2: 
             coe2.append(silcoe[i])
             datapoint2.append(i)
        elif kmeans.labels_[i]==3: 
             coe3.append(silcoe[i])
             datapoint3.append(i)
             
             
             
    ##Plot of Datapoints and Silhouette Coefficient
    trace1= Bar(x=datapoint0, y=coe0) 
    data1=[trace1]
    layout1=Layout(
            showlegend=False,
            height=600,
            width=1000,
            barmode='stack',
            xaxis=XAxis(title='Data points of first cluster'),
            yaxis=YAxis(title='Silhouette Coefficient'),
            title='Datapoints of first cluster vs Silhouette Coefficient Run '+ str(h+1))
    
    fig1 = dict( data=data1, layout=layout1) 
    name='Silhouette_Coefficients for cluster 1_' + str(h) + '.html'
    plot(fig1,filename=name)
    
    trace2= Bar(x=datapoint1, y=coe1) 
    data2=[trace2]
    layout2=Layout(
            showlegend=False,
            height=600,
            width=1000,
            barmode='stack',
            xaxis=XAxis(title='Data points of first cluster'),
            yaxis=YAxis(title='Silhouette Coefficient'),
            title='Datapoints of second cluster vs Silhouette Coefficient Run '+ str(h+1))
    
    fig2 = dict( data=data2, layout=layout2) 
    name='Silhouette_Coefficients for cluster 2_' + str(h) + '.html'
    plot(fig2,filename=name)
    
    trace3= Bar(x=datapoint2, y=coe2) 
    data3=[trace3]
    layout3=Layout(
            showlegend=False,
            height=600,
            width=1000,
            barmode='stack',
            xaxis=XAxis(title='Data points of first cluster'),
            yaxis=YAxis(title='Silhouette Coefficient'),
            title='Datapoints of third cluster vs Silhouette Coefficient Run '+ str(h+1))
    
    fig3 = dict( data=data3, layout=layout3) 
    name='Silhouette_Coefficients for cluster 3_' + str(h) + '.html'
    plot(fig3,filename=name)
    
    trace4= Bar(x=datapoint3, y=coe3) 
    data4=[trace4]
    layout4=Layout(
            showlegend=False,
            height=600,
            width=1000,
            barmode='stack',
            xaxis=XAxis(title='Data points of first cluster'),
            yaxis=YAxis(title='Silhouette Coefficient'),
            title='Datapoints of fourth cluster vs Silhouette Coefficient Run '+ str(h+1))
    
    fig4 = dict( data=data4, layout=layout4) 
    name='Silhouette_Coefficients for cluster 4_' + str(h) + '.html'
    plot(fig4,filename=name)
             
    print("Average Silhouette coefficient for the Run # :", h+1)
    print("Average Silhouette coefficient for the cluster 0 is :",np.mean(coe0));
    print("Average Silhouette coefficient for the cluster 1 is :",np.mean(coe1));
    print("Average Silhouette coefficient for the cluster 2 is :",np.mean(coe2));
    print("Average Silhouette coefficient for the cluster 3 is :",np.mean(coe3));
             
    cntN0 = cntR0 = cntN1 = cntR1 = cntN2 = cntR2 = cntN3 = cntR3 = 0
    for i in range(len(kmeans.labels_)): 
        if kmeans.labels_[i]==0: 
             if(Data1_wpbc_new.iloc[i,0] == 'N'):
                 cntN0 += 1
             else:
                 cntR0 += 1
        elif kmeans.labels_[i]==1: 
             if(Data1_wpbc_new.iloc[i,0] == 'N'):
                 cntN1 += 1
             else:
                 cntR1 += 1
        elif kmeans.labels_[i]==2: 
             if(Data1_wpbc_new.iloc[i,0] == 'N'):
                 cntN2 += 1
             else:
                 cntR2 += 1
        elif kmeans.labels_[i]==3: 
             if(Data1_wpbc_new.iloc[i,0] == 'N'):
                 cntN3 += 1
             else:
                 cntR3 += 1
    
    if(cntN0 > cntR0):
        ClassLabel0 = 'N'
        fr0 = cntN0 / (cntN0 + cntR0)
    else:
        ClassLabel0 = 'R'
        fr0 = cntR0 / (cntN0 + cntR0)
    if(cntN1 > cntR1):
        ClassLabel1 = 'N'
        fr1 = cntN1 / (cntN1 + cntR1)
    else:
        ClassLabel1 = 'R'
        fr1 = cntR1 / (cntN1 + cntR1)
    if(cntN2 > cntR2):
        ClassLabel2 = 'N'
        fr2 = cntN2 / (cntN2 + cntR2)
    else:
        ClassLabel2 = 'R'
        fr2 = cntR2 / (cntN2 + cntR2)
    if(cntN3 > cntR3):
        ClassLabel3 = 'N'
        fr3 = cntN3 / (cntN3 + cntR3)
    else:
        ClassLabel3 = 'R'
        fr3 = cntR3 / (cntN3 + cntR3)
        
    print('Class Label of cluster 0',ClassLabel0)
    print('Class Label of cluster 1',ClassLabel1)
    print('Class Label of cluster 2',ClassLabel2)
    print('Class Label of cluster 3',ClassLabel3)
    classLabel=[]
    classLabel.append(ClassLabel0)
    classLabel.append(ClassLabel1)
    classLabel.append(ClassLabel2)
    classLabel.append(ClassLabel3)
    
    print('Fraction of points of cluster 0',fr0)
    print('Fraction of points of cluster 1',fr1)
    print('Fraction of points of cluster 2',fr2)
    print('Fraction of points of cluster 3',fr3)
      
    #def pred_SSE(X, label, center):
    label = kmeans.labels_
    center = kmeans.cluster_centers_
    pred_label=[]
    for i in range(len(label)):
        eucl=[]  
        SSE0 = SSE1 = SSE2 = SSE3 = 0
        for j in range(X.shape[1]):
            SSE0 += (float(X.iloc[i,j]) - center[0][j])**2
            SSE0 = np.sqrt(SSE0)
      
            SSE1 += (float(X.iloc[i,j]) - center[1][j])**2
            SSE1 = np.sqrt(SSE1)
     
            SSE2 += (float(X.iloc[i,j]) - center[2][j])**2
            SSE2 = np.sqrt(SSE2)
    
            SSE3 += (float(X.iloc[i,j]) - center[3][j])**2
            SSE3 = np.sqrt(SSE3)
        eucl.append(SSE0)    
        eucl.append(SSE1)
        eucl.append(SSE2)
        eucl.append(SSE3)
        
        pred_label.append(classLabel[eucl.index(min(eucl))])
    conf_matrix = confusion_matrix(Y, pred_label);
      
        
    accuracy=accuracy_score(Y, pred_label)
    recallN=recall_score(Y, pred_label,average='macro', labels=['N'])
    recallR=recall_score(Y, pred_label,average='macro', labels=['R'])
    precisionN=precision_score(Y, pred_label,average='macro', labels=['N'])
    precisionR=precision_score(Y, pred_label,average='macro', labels=['R'])
    print("Confusion Matix:\n");
    print(conf_matrix)
    print("Accuarcy is :",accuracy)
    print("Recall for class N is :",recallN)
    print("Precision for class N is :",precisionN) 
    print("Recall for class R is :",recallR)
    print("Precision for class R is :",precisionR) 
        
        
        
