function [Y_predict] = rathp_QDA_test(X_test, QDAmodel, numofClass)


covar_test=QDAmodel.Sigma;%storing all the parameters from QDA model
muy=QDAmodel.Mu;
Prob=QDAmodel.Pi;
[test_rows,test_col]=size(X_test);

%computing Mahalanobis distance
Dist=zeros(numofClass,test_rows);
for j=1:test_rows
for i=1:numofClass
    Dist(i,j)=0.5.*(((X_test(j,:))'-(muy(i,:))')'*(inv(covar_test{1,i}))*((X_test(j,:))'-(muy(i,:))'))+0.5*log(det(covar_test{1,i}))-log(Prob(i,1));
end
end

%Predicting the labels using the MAP rule.

[val,Y_predict]=min(abs(Dist));
Y_predict=(Y_predict)';

end
