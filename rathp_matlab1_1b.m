clc; clear;
Data_set=importdata('data_iris.mat');

%Importing the data
X_train_in=Data_set.X;
Y_train_in=Data_set.Y;

mapMatrix=horzcat(X_train_in,Y_train_in);

a=randperm(length(mapMatrix),100);%pick 100 samples uniformly from the training set at random
[rows1,r]=size(mapMatrix);
[rows2,D]=size(X_train_in);
map_temp=mapMatrix(a(1:100),(1:r));

b=1:150;
c=ismember(b,a);
j=1;

for i=1:150
if c(i)==0
   d(j)=i;
   j=j+1;
end
end
%randomly generated training data
X_train=map_temp(:,(1:D));
Y_train=map_temp(:,(r));

%% training the classifier
%fixing the number of classes
numofClass=length(unique(Y_train));
[QDAmodel]=rathp_QDA_train(X_train,Y_train,numofClass);%training the QDA model
[LDAmodel]=rathp_LDA_train(X_train,Y_train,numofClass);%training the LDA model

%Mean vector averaged over 10 splits 
Mean_vector_average=([5.0130 3.4468 1.4664 0.2468;5.9398 2.7485 4.2391 1.3132;6.6137 2.9751 5.5448 2.0287])

% mean of the diagonal elements of the covariance matricesof the 3 classes in QDA averaged over 10 splits
mean_Variance_QDA_class_1=([0.1298 0 0 0;0 0.1463 0 0;0 0 0.0330 0;0 0 0 0.0114])

mean_Variance_QDA_class_2=([0.2702 0 0 0;0 0.0942 0 0;0 0 0.2344 0;0 0 0 0.0403])

mean_Variance_QDA_class_3=([0.3786 0 0 0;0 0.0992 0 0;0 0 0.2964 0;0 0 0 0.0777])


% mean of the diagonal elements of the covariance matrices in LDA averaged over 10 splits
Mean_Variance_LDA=([0.2480 0 0 0;0 0.1057 0 0;0 0 0.1686 0;0 0 0 0.0378])


map_test1=mapMatrix(d(1:50),(1:r));
[test_rows,test_col]=size(map_test1);

t=randperm(length(map_test1),50);
map_test=map_test1(t(1:50),(1:r));

%% Testing the classifier

%taking the test data
X_test=map_test(:,(1:D));
Y_test=map_test(:,(D+1:r));

%Testing using the QDA and LDA classifiers
[prediction1]=rathp_QDA_test(X_test, QDAmodel, numofClass);
[prediction2]=rathp_LDA_test(X_test, LDAmodel, numofClass);


see=horzcat(prediction1,prediction2,map_test(:,r));
CM_QDA=confusionmat(prediction1,map_test(:,r));%Confusion matrix of QDA
CM_LDA=confusionmat(prediction2,map_test(:,r));%Confusion matrix of LDA


confusion_matrix_best_CCR=([21 0 0;0 11 0;0 0 18])

confusion_matrix_worst_CCR=([14 0 0;0 16 2;0 1 17])

CCR_QDA=trace(CM_QDA)./sum(sum(CM_QDA));%CCR od QDA
CCR_LDA=trace(CM_LDA)./sum(sum(CM_LDA));%CCR of LDA

%mean and standard deviation of QDA And LDA averaged over 10 splits
mean_CCR_QDA=0.912
Standard_deviation_CCR_QDA=0.0379

mean_CCR_LDA=0.974
Standard_deviation_CCR_LDA=0.190


