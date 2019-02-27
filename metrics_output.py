# -*- coding: utf-8 -*-
"""
Created on Thu Nov  1 20:12:33 2018

@author: Administrator
"""
import numpy as np
from PIL import Image

# Create running variables
N_CORRECT = 0
N_ITEMS_SEEN = 0
TP=0
TN=0
FP=0
FN=0

def reset_running_variables():
    """ Resets the previous values of running variables to zero """
    global N_CORRECT, N_ITEMS_SEEN, TP, TN, FP, FN
    N_CORRECT = 0
    N_ITEMS_SEEN = 0
    TP=0
    TN=0
    FP=0
    FN=0
def update_running_variables(labs, preds):
    global N_CORRECT, N_ITEMS_SEEN
    N_CORRECT += (labs == preds).sum()
    N_ITEMS_SEEN += labs.size
def calculate_accuracy():
    global N_CORRECT, N_ITEMS_SEEN
    return float(N_CORRECT) / N_ITEMS_SEEN
def classification_standard(GT_num_ab,Pred_num_ab,width,threshold):
    global TP, TN, FP, FN
    if (GT_num_ab>0)  &  (Pred_num_ab>0):
        TP=TP+1
    if (GT_num_ab>0 ) & (Pred_num_ab==0):
        FN=FN+1
    if (GT_num_ab==0) & (Pred_num_ab==0):
        TN=TN+1
    if (GT_num_ab==0) &  (Pred_num_ab>0):
            
        if Pred_num_ab> width*width*threshold:
            FP=FP+1          
        else:
            TN=TN+1
    return TP,TN,FP,FN
def countfinalresults(path,threshold,runtimes):
    #print("********************************************************************************************")
    #print("*If the abnormal area is lager than %s ,then it would be classified as an abnormal image*" % str(threshold))
    
    accuracy=0
    label_0_acc=0
    label_1_acc=0
    label_2_acc=0
    label_1_count=0
    num_array=np.zeros([3,3])

    for i in range(runtimes):
        GT = Image.open(path+'gt_'+str(i)+'_0.png')
        Pred = Image.open(path+'pred_'+str(i)+'_0.png')
        #GT = Image.open(path+'gt_'+str(i)+'.png')
        #Pred = Image.open(path+'pred_'+str(i)+'.png')
        labels = np.array(GT)
        predictions = np.array(Pred)
        width=np.shape(labels)[1]

        pred_num_0=np.sum(predictions==0)
        pred_num_1=np.sum(predictions==1)
        pred_num_2=np.sum(predictions==2)
        
        GT_num_0=np.sum(labels==0)
        GT_num_1=np.sum(labels==1)
        GT_num_2=np.sum(labels==2)
        
        index0 = np.where(labels==0) 
        index1 = np.where(labels==1) 
        index2 = np.where(labels==2) 

        num00=np.sum(predictions[index0]==0)
        num10=np.sum(predictions[index0]==1)
        num20=np.sum(predictions[index0]==2)

        num01=np.sum(predictions[index1]==0)
        num11=np.sum(predictions[index1]==1)
        num21=np.sum(predictions[index1]==2)

        num02=np.sum(predictions[index2]==0)
        num12=np.sum(predictions[index2]==1)
        num22=np.sum(predictions[index2]==2)

        num_array[0,0]=num_array[0,0]+num00
        num_array[1,0]=num_array[1,0]+num10
        num_array[2,0]=num_array[2,0]+num20
        num_array[0,1]=num_array[0,1]+num01
        num_array[1,1]=num_array[1,1]+num11
        num_array[2,1]=num_array[2,1]+num21
        num_array[0,2]=num_array[0,2]+num02
        num_array[1,2]=num_array[1,2]+num12
        num_array[2,2]=num_array[2,2]+num22


        update_running_variables(labs=labels, preds=predictions)
        acc = calculate_accuracy()
        accuracy=acc+accuracy
        TP,TN,FP,FN=classification_standard(GT_num_1,pred_num_1,width,threshold)

        #print("----image {} accuracy: {}".format(i, acc))  
        acc0=float(num00)/GT_num_0
        acc2=float(num22)/GT_num_2
        label_0_acc=label_0_acc+acc0
        label_2_acc=label_2_acc+acc2
        #print("label {} accuracy: {}".format(0,acc0 ))        
        if GT_num_1>0:
            acc1=float(num11)/GT_num_1
            label_1_count=label_1_count+1
            #print("label {} accuracy: {}".format(1,acc1))
            label_1_acc=label_1_acc+acc1                
        #print("label {} accuracy: {}".format(2,acc2 ))
    ###########################################################################
    #print("Segmentation evaluation criteria*****************")
    PA=accuracy/runtimes
    print("      Pixel Accuracy (PA):",PA)
    if label_1_count>0:
        MPA=(label_0_acc/runtimes+label_2_acc/runtimes+label_1_acc/label_1_count)/3
        #print("Mean Pixel Accuracy (MPA):",MPA)
        acc1=label_1_acc/label_1_count
        #print("-------------label_1  acc:",acc1)
        
    else:
        MPA=((label_0_acc+label_2_acc/runtimes)/3)
        #print("Mean Pixel Accuracy (MPA):",MPA)
    acc0=label_0_acc/runtimes
    acc2=label_2_acc/runtimes
    #print("-------------label_0  acc:",acc0)
    #print("-------------label_2  acc:",acc2)
    #print(num_array)

    MIoU_1=float(num_array[1,1])/(num_array[0,1]+num_array[2,1]+num_array[1,0]+num_array[1,2]+num_array[1,1])
    MIoU_0=float(num_array[0,0])/(num_array[1,0]+num_array[2,0]+num_array[0,1]+num_array[0,2]+num_array[0,0])
    MIoU_2=float(num_array[2,2])/(num_array[2,0]+num_array[2,1]+num_array[0,2]+num_array[1,2]+num_array[2,2])
    MIoU=(MIoU_0+MIoU_1+MIoU_2)/3
    #print("==============MIoU:",MIoU)
    #print("==============MIoU_0:",MIoU_0)
    print("==============MIoU_1:",MIoU_1)
    #print("==============MIoU_2:",MIoU_2)

    MIoU_ab=float(num_array[1,1])/(num_array[2,1]+num_array[1,2]+num_array[1,1])
    MIoU_no=float(num_array[2,2])/(num_array[2,1]+num_array[1,2]+num_array[2,2])
    MIoU_class2=(MIoU_ab+MIoU_no)/2
    #print("==============MIoU_class2:",MIoU_class2)
    print("==============MIoU_ab:",MIoU_ab)
    #print("==============MIoU_no:",MIoU_no)

    ###########################################################################
    
    #print("\nClassification evaluation criteria*************")
    #print("TP {} TN {} FP {} FN {}".format(TP,TN,FP,FN))
    if (TP+FN+TN+FP)>0:
        Accuracy=float(TP+TN)/((TP+FN+TN+FP))
        #print("Accuracy:",Accuracy)
    Specificity=0
    if (TN+FP)>0:
        Specificity=TN/float(TN+FP)
        #print("Specificity:",Specificity)
    if (TP+FN)>0:
        Sensitivity=TP/float(TP+FN)
        #print("Sensitivity:",Sensitivity)


    print("********************************************************************************************")
    return PA,MPA,acc0,acc1,acc2,MIoU,MIoU_0,MIoU_1,MIoU_2,Accuracy,Specificity,Sensitivity,TP,TN,FP,FN,MIoU_class2,MIoU_ab,MIoU_no
