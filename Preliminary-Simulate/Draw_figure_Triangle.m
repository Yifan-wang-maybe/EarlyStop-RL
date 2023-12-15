clear;
clc;
quantnum=100;
load Example_Results/D_drop_yifan_P_8-6.mat
V=D;
%V = squeeze(V(:,:,:,:));

for i=4:1:30

    fig = figure;
    imagesc([0:1/(quantnum-1):1],[0:1/(quantnum-1):1],V(:,:,i,1));
    set(gca,'Xtick',[0,0.2,0.4,0.6,0.8,1]);
    set(gca,'Ytick',[0,0.2,0.4,0.6,0.8,1]);
    set(gca,'YDir','normal');  
    ylabel('Belief $\pi_N$','Fontname','Times New Roman','Fontsize',24,'Interpreter','latex')
    xlabel('Belief $\pi_P$','Fontname','Times New Roman','Fontsize',24,'Interpreter','latex')
    %title('Optimal Action','Fontname','Times New Roman','Fontsize',16)
    title('Function s','Fontname','Times New Roman','Fontsize',16)
    colorbar

%     F=getframe(fig);
%     I=frame2im(F);
%     [I,map]=rgb2ind(I,256);
%     if(i==1)
%         imwrite(I,map,'movefig.gif','DelayTime',0.1,'LoopCount',Inf)
%     else
%         imwrite(I,map,'movefig.gif','WriteMode','append','DelayTime',0.1)    
%     end
end



% clear;
% clc;
% 
% 
% 
% quantnum=100;
% load D_drop_yifan_P.mat
% V=D;
% 
% 
% for i=10:10:10
%     V_new = squeeze(V(:,:,15,:));
%     fig = figure;
%     imagesc(V_new(:,:,1));
%     set(gca,'Xtick',[0,8,16,24,32,40]);
%     set(gca,'Ytick',[0,20,40,60,80,100]);
%     set(gca,'YDir','normal');  
%     ylabel('Belief $\pi_N$','Fontname','Times New Roman','Fontsize',24,'Interpreter','latex')
%     xlabel('Diameter','Fontname','Times New Roman','Fontsize',24,'Interpreter','latex')
%     title('Optimal Action','Fontname','Times New Roman','Fontsize',16)
%     %title('Function s','Fontname','Times New Roman','Fontsize',16)
%     colorbar
%     clim([0 3])
% 
%     
%     %F=getframe(fig);
%     %I=frame2im(F);
%     %[I,map]=rgb2ind(I,256);
%     %if(i==1)
%     %    imwrite(I,map,'movefig.gif','DelayTime',0.1,'LoopCount',Inf)
%     %else
%     %    imwrite(I,map,'movefig.gif','WriteMode','append','DelayTime',0.1)    
%     %end
% end




















% v = VideoWriter('Noiseless.avi');
% for frame = 3:32
%      
%          
%      F = getframe(fig);
%      [X, Map] = frame2im(F);
%      writeVideo(v,X)
%  end
% % 


% fig = figure;
% imagesc([0:1/(quantnum-1):1],[0:1/(quantnum-1):1],V(:,:,10,1));
% set(gca,'Xtick',[0,0.2,0.4,0.6,0.8,1]);
% set(gca,'Ytick',[0,0.2,0.4,0.6,0.8,1]);
% ylabel('Belief $\pi_N$','Fontname','Times New Roman','Fontsize',24,'Interpreter','latex')
% xlabel('Belief $\pi_P$','Fontname','Times New Roman','Fontsize',24,'Interpreter','latex')
% title('Optimal','Fontname','Times New Roman','Fontsize',16)
% colorbar









% 
% open(v)
% %J1 = Ja{1};
% %J2 = Ja{2};
% %J3 = Ja{3};
% for frame = 1:1
%     fig = figure;
%     imagesc([0:1/(quantnum-1):1],[0:1/(quantnum-1):1],V(:,:,10,1));
%     set(gca,'Xtick',[0,0.2,0.4,0.6,0.8,1]);
%     set(gca,'Ytick',[0,0.2,0.4,0.6,0.8,1]);
%     ylabel('Belief $\pi_N$','Fontname','Times New Roman','Fontsize',24,'Interpreter','latex')
%     xlabel('Belief $\pi_P$','Fontname','Times New Roman','Fontsize',24,'Interpreter','latex')
%     %title('Optimal','Fontname','Times New Roman','Fontsize',16)
%     colorbar
%         
%     F = getframe(fig);
%     [X, Map] = frame2im(F);
%     writeVideo(v,X)
% end
% 
% load D_drop_yifan_21.mat

% fig = figure;
% imagesc([0:1/(quantnum-1):1],[0:1/(quantnum-1):1],J1(:,:,10,1));
% set(gca,'Xtick',[0,0.2,0.4,0.6,0.8,1]);
% set(gca,'Ytick',[0,0.2,0.4,0.6,0.8,1]);
% ylabel('Belief $\pi_N$','Fontname','Times New Roman','Fontsize',24,'Interpreter','latex')
% xlabel('Belief $\pi_P$','Fontname','Times New Roman','Fontsize',24,'Interpreter','latex')
% %title('Optimal','Fontname','Times New Roman','Fontsize',16)
% colorbar
% 
% fig = figure;
% imagesc([0:1/(quantnum-1):1],[0:1/(quantnum-1):1],J2(:,:,10,1));
% set(gca,'Xtick',[0,0.2,0.4,0.6,0.8,1]);
% set(gca,'Ytick',[0,0.2,0.4,0.6,0.8,1]);
% ylabel('Belief $\pi_N$','Fontname','Times New Roman','Fontsize',24,'Interpreter','latex')
% xlabel('Belief $\pi_P$','Fontname','Times New Roman','Fontsize',24,'Interpreter','latex')
% %title('Optimal','Fontname','Times New Roman','Fontsize',16)
% colorbar
% 
% fig = figure;
% imagesc([0:1/(quantnum-1):1],[0:1/(quantnum-1):1],J3(:,:,10,1));
% set(gca,'Xtick',[0,0.2,0.4,0.6,0.8,1]);
% set(gca,'Ytick',[0,0.2,0.4,0.6,0.8,1]);
% ylabel('Belief $\pi_N$','Fontname','Times New Roman','Fontsize',24,'Interpreter','latex')
% xlabel('Belief $\pi_P$','Fontname','Times New Roman','Fontsize',24,'Interpreter','latex')
% %title('Optimal','Fontname','Times New Roman','Fontsize',16)
% colorbar
% 
% close(v)