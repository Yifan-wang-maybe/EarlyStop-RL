clc;
clear;
clear all;

load Normalize_matrix_E.mat
load Normalize_matrix_N.mat
load Normalize_matrix_P.mat


% State transition probability N
%         N   E   P 
Lamba = [0.5,0.1,0.4];   

% Action space [CT(1),CT(2),CT(3), N, P ]
A = 5;                  

% Cost
% Diagnosis Cost
cf = [5,2.1,5,2.1];  
% Inter-step Cost
ca = [1,1,1];
cm = 1;  %%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Observation space
ZZ = 40;           % Observation
Time = 6;
% quantnum
quantnum=100;


%% Optimal Policy
V=zeros(quantnum,quantnum,ZZ,Time);
Ja=cell(A,1);
for adx=1:A
    Ja{adx}=zeros(quantnum,quantnum,ZZ,Time);
end
D=zeros(quantnum,quantnum,ZZ,Time);
Vpre=ones(quantnum,quantnum,ZZ,Time);
count=0;

epoch = 0;
Difference = abs(V-Vpre);
while sum(Difference(:))>1
    epoch = epoch + 1
    Vpre=V;
    for idx_N = 1:quantnum  % 1 : N
    for idx_P = 1:quantnum-idx_N  % 1 : N - ind_X
        pi_N=idx2pi(idx_N,quantnum);  % pi=(idx-0.5)/quantnum;
        pi_P=idx2pi(idx_P,quantnum);  % pi=(idx-0.5)/quantnum;
        
        for z1 = 4:ZZ
        for T = 1:Time

            z_current = z1;
            J=zeros(A,1);
       
            for adx=1:A            
                if adx == A-1
                    J(adx) = cf(1)*pi_P + cf(2)*(1-pi_N-pi_P);
                elseif adx == A
                    J(adx) = cf(3)*pi_N + cf(4)*(1-pi_N-pi_P);
                else
                    J(adx)= ca(adx) + cm * loss_Geo(z_current,T) * pi_P * adx;
                    
                    for ddx1=3:ZZ

                        z_next = ddx1;

                        Alpha1 = Prob_alpha(z_current,z_next,adx,Normalize_matrix_E);
                        Beta1 = Prob_beta(z_current,z_next,adx,Normalize_matrix_N);
                        Gamma1 = Prob_gamma(z_current,z_next,adx,Normalize_matrix_P);

                        ALPHA = Alpha1;
                        BATE = Beta1;
                        GAMMA = Gamma1;

                        [pi_new_N,pi_new_P,~] = Piupdate(ALPHA,BATE,GAMMA,Lamba,pi_N,pi_P);
                                               
                        SIGMA = (1-pi_P-pi_N)*Alpha1 + pi_N * Beta1 + pi_P*Gamma1;
                        
                        if T == Time
                            J(adx) = J(adx) + 0.85*SIGMA*Vpre(pi2idx(pi_new_N,quantnum),pi2idx(pi_new_P,quantnum),z_next,T);
                        else                     
                            J(adx) = J(adx) + 0.85*SIGMA*Vpre(pi2idx(pi_new_N,quantnum),pi2idx(pi_new_P,quantnum),z_next,min(T+adx,Time));
                        end
                    end
                end
                Ja{adx}(idx_N,idx_P,z_current,T)=J(adx);
                
            end
            
            [V(idx_N,idx_P,z_current,T),D(idx_N,idx_P,z_current,T)]= min(J(1:end));
            
            
            
        end
        end 
        
    end
    end
    
    Difference = abs(V-Vpre);
    error = sum(Difference(:))
    
    %count=count+1;
    %if count==-1
    %    break;
    %end
    
    save("D_drop_yifan_P.mat",'D')
    save("V_drop_yifan_P.mat",'V')

end

%V_prop=V;
%D_prop=D;



function Alpha = Prob_alpha(z,z_next,action,Normalize_matrix_E)

    Alpha = normpdf(z_next,z,action); 
    Alpha = Alpha/Normalize_matrix_E(z,action);
 

end

function Beta = Prob_beta(z,z_next,action,Normalize_matrix_N)

    Beta = normpdf(z_next,z,2*action); 
    
    Beta = Beta/Normalize_matrix_N(z,action);
    
end

function Gamma = Prob_gamma(z,z_next,action,Normalize_matrix_P)

    for i=1:action*2
        r = 0.034;
        V = (z*z*z*pi)/6;
        V = V + V*13*r*exp(-1*r);
        V = V/(pi/6);
        k = power(V,1/3);
        z = k;
    end
    
    Gamma = normpdf(z_next,k,3);
    k = round(k);
    Gamma = Gamma/Normalize_matrix_P(min(k,80),action);
    
end

function k=loss_Geo(z,time)
    
    r = 0.034;

    V = (z*z*z*pi)/6;
    V = V + V*13*r*exp(-1*r);
    V = V/(pi/6);
    V = power(V,1/3);

    dv = V-z;
    k = dv*time/5;

end


function [Pi_N,Pi_P,All]=Piupdate(Alpha,Beta,Gamma,Lamba,Pi_N,Pi_P)
  
    Pi_N_hat = Pi_N + Lamba(1)*(1-Pi_N-Pi_P);
    Pi_P_hat = Pi_P + Lamba(3)*(1-Pi_N-Pi_P);
   
    All = (1-Pi_P_hat-Pi_N_hat)*Alpha + Pi_N_hat * Beta + Pi_P_hat*Gamma;
   
    Pi_N = Pi_N_hat*Beta/All;
    Pi_P = Pi_P_hat*Gamma/All;

end


function pi=idx2pi(idx,quantnum)
    pi=(idx-0.5)/quantnum;
end

function idx=pi2idx(pi,quantnum)
    idx=ceil(pi*quantnum)+1;
    if idx>quantnum
        idx = quantnum;
    end
end
