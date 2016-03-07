function [QDAmodel] = rathp_QDA_train(X_train,Y_train,numofClass)


mapMatrix=horzcat(X_train,Y_train);
count=zeros(numofClass,1);
[rows,D]=size(X_train);
[rows,r]=size(mapMatrix);

%finding mean vector
labels=unique(Y_train);
numofClass=length(labels);
add=zeros(numofClass,D);
Mu=zeros(numofClass,D);
count=zeros(numofClass,1);

for k=1:numofClass
for i=1:rows
   if mapMatrix(i,D+1)==labels(k)
       count(k)=count(k)+1;
       for j=1:D
           add(k,j)=add(k,j)+X_train(i,j);
           Mu(k,j)=add(k,j)./count(k);
       end
   end
end
end

x_sep=cell(numofClass,1);

 for k=1:numofClass
 temp=zeros(count(k),1);
 temp=x_sep{k,1};
 j=1;
 for i=1:rows
       if mapMatrix(i,D+1)==labels(k);
       temp(j,:)=mapMatrix(i,(1:D));
       j=j+1;
    end
 end
    x_sep{k,1}=temp;
end

Pi=zeros(numofClass,1);


for i=1:numofClass
Pi(i)=count(i)./sum(count);
end

%Finding the Covariance matrices and storing them in cells

Sigma_t=cell(numofClass,1);

for j=1:numofClass
Sigma_t{j}=cov(x_sep{j});
end
Sigma=(Sigma_t)';

%returning the MU SIGMA and PI

QDAmodel=struct('Mu',Mu,'Sigma',{Sigma},'Pi',Pi);
end

