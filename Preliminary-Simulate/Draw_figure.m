clear;
clc;
quantnum=100;
load Example_Results/D_drop_yifan_P_8-6.mat
V=D;
%V = squeeze(V(:,:,:,:));

d = size(V);
V_tri = zeros(d);

B_points_X = csvread('Test_set_location/Testset_location_B_X.csv');
B_points_Y = csvread('Test_set_location/Testset_location_B_Y.csv');
M_points_X = csvread('Test_set_location/Testset_location_M_X.csv');
M_points_Y = csvread('Test_set_location/Testset_location_M_Y.csv');

[BXX,BYY] = To_Tri(B_points_X,B_points_Y);
[MXX,MYY] = To_Tri(M_points_X,M_points_Y);

for i = 1:1:d(3)
    for j = 1:1:d(4)
        for x = 1:1:d(1)
            for y = 1:1:d(2)-x
                V_tri(round(x+y*cos(pi/3)),round(y*sin(pi/3)),i,j) = V(x,y,i,j);
            end
        end
       
    end
end


for i = 10:1:40
    fig = figure;
    imagesc([0:1/(quantnum-1):1],[0:1/(quantnum-1):1],permute(V_tri(:,:,i,1),[2,1,3,4]));
    hold on
    scatter(BXX/100,BYY/100,20,"red",'filled')
    hold on
    scatter(MXX/100,MYY/100,20,"green",'filled')
    hold on
    set(gca,'Xtick',[0,0.2,0.4,0.6,0.8,1]);
    set(gca,'Ytick',[0,0.2,0.4,0.6,0.8,1]);
    set(gca,'YDir','normal');  
    ylabel('Belief $\pi_N$','Fontname','Times New Roman','Fontsize',24,'Interpreter','latex')
    xlabel('Belief $\pi_P$','Fontname','Times New Roman','Fontsize',24,'Interpreter','latex')
    title('Optimal Action','Fontname','Times New Roman','Fontsize',16)
    title('Function s','Fontname','Times New Roman','Fontsize',16)
    colorbar

end



function [XT,YT] = To_Tri(XX,YY)
    number = length(XX);
    XT = zeros(number);
    YT = zeros(number);
    for i = 1:1:number
        xx = XX(i);
        yy = YY(i);
        
        XT(i) = round(xx+yy*cos(pi/3));
        YT(i) = round(yy*sin(pi/3));
    end
end






