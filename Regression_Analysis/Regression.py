# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 14:21:13 2018

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

###############################################################################                      
                       
Data1_wpbc = pd.read_csv('wpbc.csv')
Data1_wpbc_new = Data1_wpbc.drop(['ID number', 'Lymph node status'], axis = 1) 


def digraph_tree(graphNodes, name):
        
    u = Digraph('unix', filename=name)
    u.attr(size='6,6')
    u.node_attr.update(color='lightblue2', style='filled')
    for i in range(0,len(graphNodes),2):
        u.edge(graphNodes[i], graphNodes[i+1])      
    u.view() 
       
train ,test = train_test_split(Data1_wpbc_new,test_size=0.33, stratify=Data1_wpbc_new['Outcome'])
_data = train.values[:,2:32]
_class = train.values[:,0]
   
_data_test = test.values[:,2:32]
_class_test = test.values[:,0]

feature_columns = ['radius1',	'texture1',	'perimeter1',	'area1',	'smoothness1',	'compactness1',	'concavity1',	'concave points 1',	'symmetry1',	'fractal dimension1',	'radius2',	'texture2',	'perimeter2',	'area2',	'smoothness2',	'compactness2',	'concavity2',	'concave points 2',	'symmetry2',	'fractal dimension2',	'radius3',	'texture3',	'perimeter3',	'area3',	'smoothness3',	'compactness3',	'concavity3',	'concave points 3',	'symmetry3',	'fractal dimension3']
target_column = ['Outcome']
X = pd.DataFrame(data=_data, columns = ['radius1',	'texture1',	'perimeter1',	'area1',	'smoothness1',	
                                        'compactness1',	'concavity1',	'concave points 1',	'symmetry1',	
                                        'fractal dimension1',	'radius2',	'texture2',	'perimeter2',	
                                        'area2',	'smoothness2',	'compactness2',	'concavity2',	
                                        'concave points 2',	'symmetry2',	'fractal dimension2',	'radius3',	
                                        'texture3',	'perimeter3',	'area3',	'smoothness3',	'compactness3',	
                                        'concavity3',	'concave points 3',	'symmetry3',	'fractal dimension3'])
Y = pd.DataFrame(data=_class)

accuracy=[];  
precision_N=[];  
recall_N=[];
precision_R=[];  
recall_R=[];



kf = KFold(n_splits=4, shuffle=False)

for train_ids, val_ids in kf.split(X,Y):
    
    X_Train = X.iloc[train_ids]
    Y_Train = Y.iloc[train_ids]

    X_val = X.iloc[val_ids]
    y_val = Y.iloc[val_ids]
    
    clf = tree.DecisionTreeClassifier(criterion = 'gini', max_depth=3)
    clf = clf.fit(X_Train,Y_Train)
              
y_pred_Test = clf.predict(_data_test)
y_pred_Test

acc = accuracy_score(y_pred_Test,_class_test)*100
prec_n = precision_score(y_pred_Test,_class_test,average='macro',labels=['N'])
prec_r = precision_score(y_pred_Test,_class_test,average='macro',labels=['R'])
rec_n = recall_score(y_pred_Test,_class_test,average='macro',labels=['N'])
rec_r = recall_score(y_pred_Test,_class_test,average='macro',labels=['R'])

#Visualizing the DecisionTree using Grahpviz
dot_data = tree.export_graphviz(clf, out_file=None) 
graph = graphviz.Source(dot_data) 
graph.render("Data2_loop_")

print("\nAccuracy:\n",acc)
print("Precision N:\n",prec_n)
print("Recall N:\n",rec_n)
print("Precision R:\n",prec_r)
print("Recall R:\n",rec_r)



########################################################################################################
                                  
for g in range(20):                                                             
    counter = 0
    mse_counter_w=[]
    graph_nodes=[]
    graph_nodes_number=[]
    pred_array_w=[]
    split_value_w=[]
    attr=[]
    
    x_train_q2 ,x_test_q2 = train_test_split(Data1_wpbc,test_size=0.33)
    x_train_q2_new = x_train_q2.drop(['ID number'], axis = 1)
    x_test_q2_new = x_test_q2.drop(['ID number'], axis = 1)
    
    y_train = x_train_q2_new.values[:,1]
    Y_test = x_test_q2_new.values[:,1]
    
    y_train_Reg = pd.DataFrame(data=y_train, columns=['time'])
    Y_Test_Reg = pd.DataFrame(data=Y_test, columns=['time'])
    
    avg_time_test = np.mean(y_train_Reg['time'])
    mse_test = ((avg_time_test - Y_Test_Reg['time']) ** 2).mean()
    
    def Reg_Tree(x_train_q2, location, ctr, ctrr):   
        
        d={}
        
        X_Train_Reg = pd.DataFrame(data=d)
        Y_Train_Reg = pd.DataFrame(data=d)
        
        X_Reg = pd.DataFrame(data=d)
        Y_Reg = pd.DataFrame(data=d)
        
            
        X_Reg = x_train_q2.values[:,2:32]
        Y_Reg = x_train_q2.values[:,1]
        
        X_Train_Reg = pd.DataFrame(data=X_Reg, columns=['radius1',	'texture1',	'perimeter1',	'area1',	'smoothness1',	'compactness1',	'concavity1',	'concave points 1',	'symmetry1',	'fractal dimension1',	'radius2',	'texture2',	'perimeter2',	'area2',	'smoothness2',	'compactness2',	'concavity2',	'concave points 2',	'symmetry2',	'fractal dimension2',	'radius3',	'texture3',	'perimeter3',	'area3',	'smoothness3',	'compactness3',	'concavity3',	'concave points 3',	'symmetry3',	'fractal dimension3'])
        Y_Train_Reg = pd.DataFrame(data=Y_Reg, columns=['time'])
        
        global counter;
        counter = counter + 1
        def mse(avg_time, Y_Train_Reg):
            return  ((avg_time - Y_Train_Reg['time']) ** 2).mean()
        
        avg_time_train = np.mean(Y_Train_Reg['time'])
        
        if Y_Train_Reg.shape[0] == 0:
            return
        
        mse_train = mse(avg_time_train, Y_Train_Reg)
        
        #print("Node type",location)
        #print("MSE_Train",mse_train)
        
        if location != 'Root':
            graph_nodes_number.append(ctrr)
            graph_nodes_number.append(counter)
            ctrr = counter
            
        flag_w = True
        mse_counter_w.append(mse_train)
        
        if(mse_train < 200):
            flag_w = False
            
        #array_node.append(counter);
        if(x_train_q2.shape[0] > 10 and flag_w == True):
            corr_matrix = x_train_q2.corr()
            #corr_matrix_time = np.sort(corr_matrix.iloc[:,1])      
            corr_matrix_time = corr_matrix.sort_values('Time',ascending=False)      
            Most_Corr_Attribute = corr_matrix_time.index.values[1]
            Most_Corr_Attribute_Median = np.median(x_train_q2[Most_Corr_Attribute])
            
            if(location == 'Root'):
                ctr=str('Node '+str(counter)+'\n'+str(Most_Corr_Attribute)+' '+str(Most_Corr_Attribute_Median)+'\nmse='+str(mse_train)+'\nsamples='+str(x_train_q2.shape[0])+'\nPrediction='+str(avg_time_train))
            else:    
                graph_nodes.append(ctr)
                graph_nodes.append(str('Node '+str(counter)+'\n'+str(Most_Corr_Attribute)+' '+str(Most_Corr_Attribute_Median)+'\nmse='+str(mse_train)+'\nsamples='+str(x_train_q2.shape[0])+'\nPrediction='+str(avg_time_train)))
                ctr=str('Node '+str(counter)+'\n'+str(Most_Corr_Attribute)+' '+str(Most_Corr_Attribute_Median)+'\nmse='+str(mse_train)+'\nsamples='+str(x_train_q2.shape[0])+'\nPrediction='+str(avg_time_train))
                    
            d={}
            df_split1 = pd.DataFrame(data=d)
            df_split2 = pd.DataFrame(data=d)
            df_split1 = x_train_q2[x_train_q2[Most_Corr_Attribute] <= Most_Corr_Attribute_Median]
            df_split2 = x_train_q2[x_train_q2[Most_Corr_Attribute] > Most_Corr_Attribute_Median]
            
            split_value_w.append(Most_Corr_Attribute_Median)
            pred_array_w.append(avg_time_train)
            attr.append(Most_Corr_Attribute)
            
            if (df_split1.shape[0] != 0):
                Reg_Tree(df_split1, "Left", ctr, ctrr)
            if (df_split1.shape[0] != 0):
                Reg_Tree(df_split2, "Right", ctr, ctrr)
                
        else:
            #print("\nNode Number:", array_node[counter-1])
            graph_nodes.append(ctr)
            graph_nodes.append(str(str('Node '+str(counter))+'\nmse='+str(mse_train)+'\nsamples='+str(x_train_q2.shape[0])+'\nPrediction='+str(avg_time_train)))
            ctr=str(str('Node '+str(counter))+'\nmse='+str(mse_train)+'\nsamples='+str(x_train_q2.shape[0])+'\nPrediction='+str(avg_time_train))
            
            split_value_w.append(-1)
            pred_array_w.append(avg_time_train)
            attr.append('-1')
            
#            print("\nMSE:",mse_train)
#            print("\nPopulation count at leaf node:",x_train_q2.shape[0])
#            print("Predicted Value:",avg_time_train)
    
    location = "Root" 
    print("#################")
    print("WPBC Dataset ",g)
    print("#################")
          
    Reg_Tree(x_train_q2_new, location, 1, 1)
    
    
    
    w, h = 3, counter;
    Matrix_w = [[0 for x in range(w)] for y in range(h)]
    matrix_counter_w = 0
    for i in range(0, len(graph_nodes_number), 2):
        if(graph_nodes_number[i] != -1):        
            Matrix_w[matrix_counter_w][0] = graph_nodes_number[i]
            graph_nodes_number[i] = -1
            Matrix_w[matrix_counter_w][1] = graph_nodes_number[i+1]
            graph_nodes_number[i+1] = -1
            for j in range(i+2, len(graph_nodes_number), 2):
                if(graph_nodes_number[j] == Matrix_w[matrix_counter_w][0]):
                    graph_nodes_number[j] = -1
                    Matrix_w[matrix_counter_w][2] = graph_nodes_number[j+1]
                    graph_nodes_number[j+1] = -1
            matrix_counter_w+=1
            
    ##################################################
                   #Prediction#
    ################################################## 
    y_pred=[]
    x_test_time_column=[]
    def patient_pred(x_test_q2):
        
        for i in range(x_test_q2.shape[0]):
            root = Matrix_w[0][0]        
            for j in range(matrix_counter_w):
                if(Matrix_w[j][0] == root):
                    left = Matrix_w[j][1]
                    right = Matrix_w[j][2]
                    #print(str(x_test_q2[attr[root - 1]].iloc[i]) + ' <= ' + str(split_value_w[root - 1]))
                    if(x_test_q2[attr[root - 1]].iloc[i] <= split_value_w[root - 1]):
                        root = left
                    else:
                        root = right
            y_pred.append(pred_array_w[root - 1])
            x_test_time_column.append(x_test_q2_new['Time'].iloc[i])
        
        
    patient_pred(x_test_q2_new)
    y_pred_test = mean_squared_error(y_pred, x_test_time_column)
    #y_pred_test = ((avg_time_test - y_pred) ** 2).mean()
    
    print(mse_test)
    print(y_pred_test)
    


    def digraph_tree(graphNodes, name):
        
        u = Digraph('unix', filename=name)
        u.attr(size='6,6')
        u.node_attr.update(color='lightblue2', style='filled')
        for p in range(0,len(graphNodes),2):
            u.edge(graphNodes[p], graphNodes[p+1])      
        u.view() 
     
    #print(graph_nodes)
    file_name = "WPBCtree"+str(g)+".gv"
    digraph_tree(graph_nodes, file_name)  
    
#    w, h = 3, counter;
#    Matrix_w = [[0 for x in range(w)] for y in range(h)]
#    matrix_counter_w = 0
#    for i in range(0, len(graph_nodes_number), 2):
#        if(graph_nodes_number[i] != -1):        
#            Matrix_w[matrix_counter_w][0] = graph_nodes_number[i]
#            graph_nodes_number[i] = -1
#            Matrix_w[matrix_counter_w][1] = graph_nodes_number[i+1]
#            graph_nodes_number[i+1] = -1
#            for j in range(i+2, len(graph_nodes_number), 2):
#                if(graph_nodes_number[j] == Matrix_w[matrix_counter_w][0]):
#                    graph_nodes_number[j] = -1
#                    Matrix_w[matrix_counter_w][2] = graph_nodes_number[j+1]
#                    graph_nodes_number[j+1] = -1
#            matrix_counter_w+=1
        

########################################################################################################

                       
counter_wine = 0
pred_array=[]
split_value=[]
mse_counter=[]
attr_wine=[]
#array_counter = 0
graph_nodes_wine=[]
graph_nodes_wine_number=[]

Data1_red = pd.read_csv('winequality_red.csv')                            
Data1_white = pd.read_csv('winequality_white.csv')  
x_train_q2_wine ,x_test_q2_wine = train_test_split(Data1_white,test_size=0.33)

y_train = x_train_q2_wine.values[:,11]
Y_test = x_test_q2_wine.values[:,11]

y_train_Reg = pd.DataFrame(data=y_train, columns=['quality'])
Y_Test_Reg = pd.DataFrame(data=Y_test, columns=['quality'])

avg_quality_test = np.mean(y_train_Reg['quality'])
mse_test_wine = ((avg_quality_test - Y_Test_Reg['quality']) ** 2).mean()

def Reg_Tree_wine(x_train_q2_wine, location_wine, ctr, ctrr):  
    d={}
    X_Train_Reg_wine = pd.DataFrame(data=d)
    Y_Train_Reg_wine = pd.DataFrame(data=d)
    
    X_Reg_wine = pd.DataFrame(data=d)
    Y_Reg_wine = pd.DataFrame(data=d)
        
    X_Reg_wine = x_train_q2_wine.values[:,0:11]
    Y_Reg_wine = x_train_q2_wine.values[:,11]
    
    X_Train_Reg_wine = pd.DataFrame(data=X_Reg_wine)
    Y_Train_Reg_wine = pd.DataFrame(data=Y_Reg_wine, columns=['quality'])
    
    global counter_wine;
    #global array_counter;
    counter_wine = counter_wine + 1
    def mse(avg_time, Y_Train_Reg_wine):
        return ((avg_time - Y_Train_Reg_wine['quality']) ** 2).mean()
    
    if Y_Train_Reg_wine.shape[0] == 0:
        return
    avg_time_train_wine = np.mean(Y_Train_Reg_wine['quality'])
    #avg_time_test_wine = np.mean(Y_Train_Reg_test_wine['quality'])
    
    mse_train_wine = mse(avg_time_train_wine, Y_Train_Reg_wine)
    #mse_test_wine = mse(avg_time_test_wine, Y_Train_Reg_test_wine)
    print("Node type",location_wine)
    print("MSE_Train",mse_train_wine)
    if location_wine != 'Root':
        graph_nodes_wine_number.append(ctrr)
        graph_nodes_wine_number.append(counter_wine)
        ctrr = counter_wine
        
        
    flag = True
    
    print("Counter ",counter_wine)
    mse_counter.append(mse_train_wine)
    if(mse_train_wine < 0.4):
        flag = False
    #array_node.append(counter);
    if(x_train_q2_wine.shape[0] > 400 and flag == True):
        
        corr_matrix = x_train_q2_wine.corr()
        #corr_matrix_time = np.sort(corr_matrix.iloc[:,1])        
        corr_matrix_time_wine = corr_matrix.sort_values('quality',ascending=False)        
        Most_Corr_Attribute_wine = corr_matrix_time_wine.index.values[11]
        Most_Corr_Attribute_Median_wine = np.median(x_train_q2_wine[Most_Corr_Attribute_wine])
        
        if(location_wine == 'Root'):
            ctr=str('Node '+str(counter_wine)+'\n'+str(Most_Corr_Attribute_wine)+' '+str(Most_Corr_Attribute_Median_wine)+'\nmse='+str(mse_train_wine)+'\nsamples='+str(x_train_q2_wine.shape[0])+'\nPrediction='+str(avg_time_train_wine))
        else:    
            graph_nodes_wine.append(ctr)
            graph_nodes_wine.append(str('Node '+str(counter_wine)+'\n'+str(Most_Corr_Attribute_wine)+' '+str(Most_Corr_Attribute_Median_wine)+'\nmse='+str(mse_train_wine)+'\nsamples='+str(x_train_q2_wine.shape[0])+'\nPrediction='+str(avg_time_train_wine)))
            ctr=str('Node '+str(counter_wine)+'\n'+str(Most_Corr_Attribute_wine)+' '+str(Most_Corr_Attribute_Median_wine)+'\nmse='+str(mse_train_wine)+'\nsamples='+str(x_train_q2_wine.shape[0])+'\nPrediction='+str(avg_time_train_wine))
            
        split_value.append(Most_Corr_Attribute_Median_wine)
        pred_array.append(avg_time_train_wine)
        attr_wine.append(Most_Corr_Attribute_wine)
        
        d={}
        df_split1_wine = pd.DataFrame(data=d)
        df_split2_wine = pd.DataFrame(data=d)
        df_split1_wine = x_train_q2_wine[x_train_q2_wine[Most_Corr_Attribute_wine] <= Most_Corr_Attribute_Median_wine]
        df_split2_wine = x_train_q2_wine[x_train_q2_wine[Most_Corr_Attribute_wine] > Most_Corr_Attribute_Median_wine]
        if (df_split1_wine.shape[0] != 0):
            Reg_Tree_wine(df_split1_wine, "Left", ctr, ctrr)
        if (df_split2_wine.shape[0] != 0):   
            Reg_Tree_wine(df_split2_wine, "Right", ctr, ctrr)
        
    
    else:
        #print("\nNode Number:", array_node[counter-1])
        graph_nodes_wine.append(ctr)
        graph_nodes_wine.append(str(str('Node '+str(counter_wine))+'\nmse='+str(mse_train_wine)+'\nsamples='+str(x_train_q2_wine.shape[0])+'\nPrediction='+str(avg_time_train_wine)))
        ctr=str(str('Node '+str(counter_wine))+'\nmse='+str(mse_train_wine)+'\nsamples='+str(x_train_q2_wine.shape[0])+'\nPrediction='+str(avg_time_train_wine))
        
        split_value.append(-1)
        pred_array.append(avg_time_train_wine)
        attr_wine.append('-1')
        
#        print("\nMSE:",mse_train_wine)
#        print("Population count at leaf node:",x_train_q2_wine.shape[0])
#        print("Predicted Value:",avg_time_train_wine)
        

location_wine = "Root"

print("\n#############################")
print("winequality_white Dataset")
print("#############################\n")
Reg_Tree_wine(x_train_q2_wine, location_wine, 1, 1) 
    
#print(graph_nodes_wine)
#print(graph_nodes_wine_number)
digraph_tree(graph_nodes_wine, 'WINEtree.gv')
    
w, h = 3, counter_wine;
Matrix = [[0 for x in range(w)] for y in range(h)]
matrix_counter = 0
for i in range(0, len(graph_nodes_wine_number), 2):
    if(graph_nodes_wine_number[i] != -1):        
        Matrix[matrix_counter][0] = graph_nodes_wine_number[i]
        graph_nodes_wine_number[i] = -1
        Matrix[matrix_counter][1] = graph_nodes_wine_number[i+1]
        graph_nodes_wine_number[i+1] = -1
        for j in range(i+2, len(graph_nodes_wine_number), 2):
            if(graph_nodes_wine_number[j] == Matrix[matrix_counter][0]):
                graph_nodes_wine_number[j] = -1
                Matrix[matrix_counter][2] = graph_nodes_wine_number[j+1]
                graph_nodes_wine_number[j+1] = -1
        matrix_counter+=1
    
 


##################################################
               #Prediction White wine#
################################################## 
y_pred_wine=[]
x_test_quality_column=[]
def patient_pred(x_test_q2):
    
    for i in range(x_test_q2.shape[0]):
        root = Matrix[0][0]        
        for j in range(matrix_counter):
            if(Matrix[j][0] == root):
                left = Matrix[j][1]
                right = Matrix[j][2]
                #print(str(x_test_q2[attr[root - 1]].iloc[i]) + ' <= ' + str(split_value_w[root - 1]))
                if(x_test_q2[attr_wine[root - 1]].iloc[i] <= split_value[root - 1]):
                    root = left
                else:
                    root = right
        y_pred_wine.append(pred_array[root - 1])
        x_test_quality_column.append(x_test_q2['quality'].iloc[i])
    
    
patient_pred(x_test_q2_wine)
y_pred_test = mean_squared_error(y_pred_wine, x_test_quality_column)
#y_pred_test = ((avg_time_test - y_pred) ** 2).mean()

print(mse_test_wine)
print(y_pred_test)

    
    
 
########################################################################################################

    
##################################################
               #Prediction Red wine#
################################################## 
y_pred_redwine=[]
x_test_quality_column_red=[]
def patient_pred_red(x_test_q2):
    
    for i in range(x_test_q2.shape[0]):
        root = Matrix[0][0]        
        for j in range(matrix_counter):
            if(Matrix[j][0] == root):
                left = Matrix[j][1]
                right = Matrix[j][2]
                #print(str(x_test_q2[attr[root - 1]].iloc[i]) + ' <= ' + str(split_value_w[root - 1]))
                if(x_test_q2[attr_wine[root - 1]].iloc[i] <= split_value[root - 1]):
                    root = left
                else:
                    root = right
        y_pred_wine.append(pred_array[root - 1])
        x_test_quality_column.append(x_test_q2['quality'].iloc[i])
    
    
patient_pred_red(Data1_red)
y_pred_test_red = mean_squared_error(y_pred_wine, x_test_quality_column)
#y_pred_test = ((avg_time_test - y_pred) ** 2).mean()

#print(mse_test_wine)
print("MSE of wine train data",mse_test_wine)
print("MSE of white wine test data",y_pred_test)
print("MSE of red wine test data",y_pred_test_red)


    
    
    
    
    
    
    
    
    
