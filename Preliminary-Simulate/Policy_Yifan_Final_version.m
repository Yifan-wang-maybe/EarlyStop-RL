clear all;


% State transition probability N
%         N   E   P 
% Lamba = [0.3,0.5,0.2];  
Lamba = [0.5,0.1,0.4];  

% Action space [CT, N, P ]
A = 3;                  

% Cost
% Diagnosis Cost
cf = [12,6.1,15,6.1];  
% Inter-step Cost
ca = [3.5];
cm = 1;  %%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Observation space
ZZ = 40;           % Observation
Time = 4;
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

iiii = 0;
Difference = abs(V-Vpre);
while sum(Difference(:))>1

    iiii = iiii + 1
    Vpre=V;
    for idx_N = 1:quantnum  % 1 : N
    for idx_P = 1:quantnum-idx_N  % 1 : N - ind_X
        pi_N=idx2pi(idx_N,quantnum);  % pi=(idx-0.5)/quantnum;
        pi_P=idx2pi(idx_P,quantnum);  % pi=(idx-0.5)/quantnum;
        
        for z1 = 3:ZZ
        for T = 1:Time

            z_current = z1;
            J=zeros(A,1);
       
            for adx=1:A            
                if adx == A-1
                    J(adx) = cf(1)*pi_P + cf(2)*(1-pi_N-pi_P);
                elseif adx == A
                    J(adx) = cf(3)*pi_N + cf(4)*(1-pi_N-pi_P);
                else
                    J(adx)= ca(adx) + cm * loss_Geo(z_current,T) * pi_P;
                    
                    
                    %J(adx)= ca(adx) + T;
                    %J(adx) = ca(adx) + T/10 + z1/2;
                    for ddx1=3:ZZ

                        z_next = ddx1;

                        Alpha1 = Prob_alpha(z_current,z_next);
                        Beta1 = Prob_beta(z_current,z_next);
                        Gamma1 = Prob_gamma(z_current,z_next);

                        ALPHA = Alpha1;
                        BATE = Beta1;
                        GAMMA = Gamma1;

                        [pi_new_N,pi_new_P,~] = Piupdate(ALPHA,BATE,GAMMA,Lamba,pi_N,pi_P);
                                               
                        SIGMA = (1-pi_P-pi_N)*Alpha1 + pi_N * Beta1 + pi_P*Gamma1;
                        
                        if T == 4
                            J(adx) = J(adx) + 0.90*SIGMA*Vpre(pi2idx(pi_new_N,quantnum),pi2idx(pi_new_P,quantnum),z_next,T);
                        else                     
                            J(adx) = J(adx) + 0.90*SIGMA*Vpre(pi2idx(pi_new_N,quantnum),pi2idx(pi_new_P,quantnum),z_next,T+1);
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
    
    
    
    count=count+1;
    if count==-1
        break;
    end
    
    Difference = abs(V-Vpre);
    stop = sum(Difference(:))
    save("D_drop_yifan_P_8-6.mat",'D')
    save("V_drop_yifan_P_8-6.mat",'V')

end

%V_prop=V;
%D_prop=D;



function Alpha = Prob_alpha(z,z_next)

    Alpha = normpdf(z_next,z,1);      
    
end

function Beta = Prob_beta(z,z_next)

    Beta = normpdf(z_next,z,3);      
    
end

function Gamma = Prob_gamma(z,z_next)
    
    r = 0.034;

    V = (z*z*z*pi)/6;
    V = V + V*13*r*exp(-1*r);
    V = V/(pi/6);
    k = power(V,1/3);


    Gamma = normpdf(z_next,k,3);      

    
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
    if idx>100
        idx = 100;
    end
end
