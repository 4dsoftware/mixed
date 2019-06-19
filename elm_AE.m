function [Y,P,TrainingTime,TestingTime,TrainingAccuracy,TestingAccuracy] = elm_AE(Tinputs,Tsinputs,NumberofHiddenNeurons,ActivationFunction,D_ratio,DB,frame)
% Input:
% Tinputs               - training data set
% Tsinputs              - testing data set
% NumberofHiddenNeurons - Number of hidden neurons assigned to the ELM
% D_ratio               - the ratio of noising features in each input frame
                        % ex: D_ratio =0.6 : 60% = gaussian noise  is added with  each input frame    
% DB                    - the power of white gaussian noise in decibels                     
% ActivationFunction    - Type of activation function:
%                           'sig' for Sigmoidal function
%                           'sin' for Sine function
%                           'hardlim' for Hardlim function
%                           'tribas' for Triangular basis function
%                           'radbas' for Radial basis function (for additive type of SLFNs instead of RBF type of SLFNs)
% frame                 - divide data into frames to add different type of noise
% Output: 
% TrainingTime          - Time (seconds) spent on training ELM
% TestingTime           - Time (seconds) spent on predicting ALL testing data
% TrainingAccuracy      - Training accuracy: RMSE
% TestingAccuracy       - Testing accuracy: RMSE
%
%
%
    %%%%    Authors:        TAREK BERGHOUT
    %%%%    BATNA 2 TECHNOLOGICAL UNIVERSITY, ALGERIA
    %%%%    EMAIL:          berghouttarek@gmail.com


%%%%%%%%%%% scale training dataset
T=Tinputs';T = scaledata(T,0,1);% memorize originale copy of the input and use it as a target
P=Tinputs';
%%%%%%%%%%% scale training dataset
TV.T=Tsinputs';TV.T = scaledata(TV.T,0,1);% memorize originale copy of the input and use it as a target
TV.P=Tsinputs';TV.P = scaledata(TV.P,0,1);% temporal input
TVT=TV.T;%save acopy as an output of the function

%%%%% in the 1st and 2nd step we will corrupte the temporal input%%%%%%%%%
PtNoise=zeros(size(P));
i=1;
while i < size(P,2)-frame
gen=randi([0,1],1,1);
PNoise=[];

%%% 1st step: generate set of indexes to set some input's values to zero later 
%%% (here we set them randomly and you can choose them by probability)%%%
[zeroind] = dividerand(size(P,1),1-D_ratio,0,D_ratio);% generate indexes
%%% 2nd step: add gaussian noise 
if gen==1
Noise=wgn(1,size(P,1),DB)';% generate white gaussian noise
else
Noise=zeros(1,size(P,1))'; 
end

for j=1:frame;%copy  noise
PNoise=[PNoise Noise];
end
if gen==1
for j=1:length(zeroind);% set to zero
    PNoise(zeroind(j),:)=0;
    P(zeroind(j),i:i+frame)=0;
end
end

PtNoise(:,i:i+frame-1)=PNoise;
i=i+frame;
end

P=P+PtNoise; % add gauussian noise (corrupte the input)
P = scaledata(P,0,1);% corrupted input
%%%%%%%%%%%%
NumberofTrainingData=size(P,2);
NumberofTestingData=size(TV.P,2);
NumberofInputNeurons=size(P,1);
%%%%%%%%%%% Calculate weights & biases
start_time_train=cputime;
%%%%%%%%%%% Random generate input weights InputWeight (w_i) and biases BiasofHiddenNeurons (b_i) of hidden neurons
InputWeight=rand(NumberofHiddenNeurons,NumberofInputNeurons)*2-1;
BiasofHiddenNeurons=rand(NumberofHiddenNeurons,1);
tempH=InputWeight*P;
%clear P;                                            %   Release input of training data 
ind=ones(1,NumberofTrainingData);
BiasMatrix=BiasofHiddenNeurons(:,ind);              %   Extend the bias matrix BiasofHiddenNeurons to match the demention of H
tempH=tempH+BiasMatrix;

%%%%%%%%%%% Calculate hidden neuron output matrix H
switch lower(ActivationFunction)
    case {'sig','sigmoid'}
        %%%%%%%% Sigmoid 
        H = 1 ./ (1 + exp(-tempH));
    case {'sin','sine'}
        %%%%%%%% Sine
        H = sin(tempH);    
    case {'hardlim'}
        %%%%%%%% Hard Limit
        H = double(hardlim(tempH));
    case {'tribas'}
        %%%%%%%% Triangular basis function
        H = tribas(tempH);
    case {'radbas'}
        %%%%%%%% Radial basis function
        H = radbas(tempH);
        %%%%%%%% More activation functions can be added here                
end
clear tempH;                                        %   Release the temparary array for calculation of hidden neuron output matrix H

%%%%%%%%%%% Calculate output weights OutputWeight (beta_i)
OutputWeight=pinv(H') * T';                            % implementation without regularization factor //refer to 2006 Neurocomputing paper
%OutputWeight=inv(eye(size(H,1))/C+H * H') * H * T';   % faster method 1 //refer to 2012 IEEE TSMC-B paper
%OutputWeight=(eye(size(H,1))/C+H * H') \ H * T';      % faster method 2 //refer to 2012 IEEE TSMC-B paper

%If you use faster methods or kernel method, PLEASE CITE in your paper properly: 

end_time_train=cputime;
TrainingTime=end_time_train-start_time_train;        %   Calculate CPU time (seconds) spent for training ELM

%%%%%%%%%%% Calculate the training accuracy
Y=(H' * OutputWeight)'    ;                          %   Y: the actual output of the corrupted training data

    TrainingAccuracy=sqrt(mse(T - Y)) ;              %   Calculate training accuracy (RMSE) 
%%%%%%%%%%%%%

%%%%% testing phase

%%%%%%%%%%% Calculate the output of testing input
start_time_test=cputime;
tempH_test=InputWeight*TV.P;
clear TV.P; %   Release input of testing data             
ind=ones(1,NumberofTestingData);
BiasMatrix=BiasofHiddenNeurons(:,ind);              %   Extend the bias matrix BiasofHiddenNeurons to match the demention of H
tempH_test=tempH_test + BiasMatrix;
switch lower(ActivationFunction)
    case {'sig','sigmoid'}
        %%%%%%%% Sigmoid 
        H_test = 1 ./ (1 + exp(-tempH_test));
    case {'sin','sine'}
        %%%%%%%% Sine
        H_test = sin(tempH_test);        
    case {'hardlim'}
        %%%%%%%% Hard Limit
        H_test = hardlim(tempH_test);        
    case {'tribas'}
        %%%%%%%% Triangular basis function
        H_test = tribas(tempH_test);        
    case {'radbas'}
        %%%%%%%% Radial basis function
        H_test = radbas(tempH_test);        
        %%%%%%%% More activation functions can be added here        
end
TY=(H_test' * OutputWeight)';                        %   TY: the actual output of the corrupted testing data
end_time_test=cputime;
TestingTime=end_time_test-start_time_test;           %   Calculate CPU time (seconds) spent by ELM predicting the whole testing data
TestingAccuracy=sqrt(mse(TV.T - TY)) ;               %   Calculate testing accuracy (RMSE) 
end