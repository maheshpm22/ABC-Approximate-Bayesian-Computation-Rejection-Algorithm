%% Question 1
clc ; clear all ; close all ; 
load('E:\PnSE_Project\Problem1\dataset1.mat')

N=10000;
 uk1=t(1,2:50);
 uk2=t(1,1:49);

 yobs=y(3:51);
pars=[];
ctr=0;
 yest=zeros(N,49);
 yest(:,1)=yobs(1);  % yobs(1) = y(3)

d_a1=0.4+sqrt(0.2)*randn(N,1);
d_b1=0.5+sqrt(0.2)*randn(N,1);
d_b2=0.3+sqrt(0.3)*randn(N,1);

for i=1:N
    for j=2:49
        yest(i,j)=d_a1(i)*yest(i,j-1)+d_b1(i)*uk1(j)-d_b2(i)*uk2(j);
    end
    if (norm(yobs-yest(i,:),2) < 200)
        
        pars=[pars; d_a1(i) d_b1(i) d_b2(i)];
        ctr=ctr+1;
    end
end
% pars is posterior distribution (distribution of parameters wrt given data)

%% Question 1 results

% load("C:\Users\Admin\Desktop\project_q1.mat"); %The values I obtained. 
figure;
histogram(pars(:,1))
title('f(a1|yn)')
figure;
histogram(pars(:,2))
title('f(b1|yn)')
figure;
histogram(pars(:,3))
title('f(b2|yn)')


A1=adtest(pars(:,1),'Distribution','norm');
B1=adtest(pars(:,2),'Distribution','norm');
B2=adtest(pars(:,3),'Distribution','norm');

point_estimates=mean(pars);
increase=100*abs(var(pars)-[0.2 0.2 0.3])./[0.2 0.2 0.3]
beta=1-exp(-info([0.4,0.5,0.3],point_estimates,[0.2,0.2,0.3],var(pars)));


%% Question 2 (Order=1) 

load('E:\PnSE_Project\Problem 2\dataset2.mat')


% Order 1
N=10000;
n=numel(data2(:,1));
pars=0.5+randn(N,3);
pstar=[0 0 0];
for i=1:N
    p=pars(i,:);
    x=zeros(n,1);
    x(1)=data2(1,2)-p(2)*data2(1,1);
    for j=2:n
        x(j)=-p(1)*x(j-1)+(p(3)-p(1)*p(2))*data2(j-1,1);
    end
    y_hat=x+p(2)*data2(:,1);
    if(norm(data2(:,2)-y_hat,2)<41)
        pstar=[pstar;p];
    end  
end

pstar=pstar(2:end,:);
A=0;

for i=1:3
A=[A adtest(pstar(:,i),'Distribution','norm')];
end

A=A(2:end);
% All are normally distributed

xfinal1=zeros(n,1);
xfinal1(1)=data2(1,2)-point1(2)*data2(1,1);
for j=2:n
    xfinal1(j)=-point1(1)*xfinal1(j-1)+(point1(3)-point1(1)*point1(2))*data2(j-1,1);
end

y_hat1=xfinal1+point1(2)*data2(:,1);

%% Question 2 (Order=2) 
% Order 2
load("C:\Users\Admin\Desktop\PnSE_Project\Problem 2\dataset2.mat")
N=10000;
n=numel(data2(:,1));
pars=0.5+randn(N,5);
pstar2=[0 0 0 0 0];

for i=1:N
    p=pars(i,:);
    x=zeros(2,n);
    x(1,1)=data2(1,2)-p(3)*data2(1,1);
    x(2,1)=x(1,1);
    for j=2:n
        x(:,j)=[-p(1) 1;-p(2) 0]*x(:,j-1)+[p(4)-p(1)*p(3);p(5)-p(2)*p(3)]*data2(j-1,1);
    end
    y_hat=transpose([1 0]*x)+p(3)*data2(:,1);
    if(norm(data2(:,2)-y_hat,2)<55)
        pstar2=[pstar2;p];
    end  
end
pstar2=pstar2(2:end,:);

A2=0;
for i=1:5
A2=[A2 adtest(pstar2(:,i),'Distribution','norm')];
end
A2=A2(2:end);

xfinal2=zeros(2,n);
xfinal2(1,1)=data2(1,2)-point2(3)*data2(1,1);
xfinal2(2,1)=xfinal2(1,1);
for j=2:n
    xfinal2(:,j)=[-point2(1) 1;-point2(2) 0]*xfinal2(:,j-1)+[point2(4)-point2(1)*point2(3);point2(5)-point2(2)*point2(3)]*data2(j-1,1);
end
y_hat2=transpose([1 0]*xfinal2)+point2(3)*data2(:,1);

%% Question 2 (Third part) 
load("C:\Users\Admin\Desktop\project_q2final.mat") % The values I obtained
% Measures for posterior 1
point1=mean(pstar);
mse1=mean((y_hat1-data2(:,2)).^2);
v1=var(pstar);
% Measures for posterior 2
point2=mean(pstar2);
mse2=mean((y_hat2-data2(:,2)).^2);
v2=var(pstar2);
% Beta information gain for parameters
info([0.5 0.5 0.5],point1,[1 1 1],v1)
info([0.5 0.5 0.5 0.5 0.5],point2,[1 1 1 1 1],v2)
pd1=fitdist(pstar(:,1),'Normal');
pd2=fitdist(pstar2(:,1),'Normal');
nll1=negloglik(pd1);
nll2=negloglik(pd2);
% AIC values for a1 for order=1, order=2
aicbic(-nll1,3)
aicbic(-nll2,5)

%% Functions
function f=info(mean1,mean2,var1,var2)
f=0.25*log(0.25*(var1./var2+var2./var1+2))+0.25*((mean1-mean2).^2./(var1.^2+var2.^2));
end