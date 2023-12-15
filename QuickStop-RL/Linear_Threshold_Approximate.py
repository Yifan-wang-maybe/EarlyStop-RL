import matplotlib.ticker as ticker
import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
import cv2
import matplotlib as mpl

from matplotlib.animation import FFMpegWriter




from sklearn.datasets import make_moons
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


def get_boundry(Data):
    X_Y = []
    label = []
    k = 25
    for x in range(100):
        for y in range(100):
            X_Y.append([x,y])

            LLable = Data[x,y,k]
            #if LLable == 2:
            #    label.append(1)
            #else:  
            #    label.append(0) 
            label.append(LLable) 
    
    #svc = LinearSVC(penalty='l2')
    svc = SVC(kernel='linear')
    svc.fit(X_Y, label)
    pre_y = svc.predict(X_Y)
    accuracy = accuracy_score(label, pre_y)
    print(accuracy)
    return svc
    

def plot_predictions(clf, axes):
    x0s = np.linspace(axes[0], axes[1], 100)
    x1s = np.linspace(axes[2], axes[3], 100)
    x0, x1 = np.meshgrid(x0s, x1s)
    X = np.c_[x0.ravel(), x1.ravel()]
    y_pred = clf.predict(X)
    y_pred = y_pred.reshape(x0.shape)
    plt.contourf(x0, x1, y_pred, cmap=plt.cm.brg, alpha=0.2)
    #plt.show()



Result = np.load('Example_Result/OUTPUTP.npy', allow_pickle = True)    
Zero_one = np.zeros((100,100,40))

for x in range(100):
    for y in range(100-x):
        for k in range(40):   ## Diameter of initial nodule ##

            reward_P = np.float(x) * 15/100 + np.float(100-x-y) *3.1/100
            reward_N = np.float(y) * 12/100 + np.float(100-x-y) *4.1/100

            dias = np.minimum(reward_P,reward_N)

            if Result[x,y,k] > dias and reward_P<reward_N:
                Zero_one[x,y,k] = 3
            elif Result[x,y,k] > dias and reward_P>=reward_N:
                Zero_one[x,y,k] = 2
            else:
                Zero_one[x,y,k] = 1
    
    
    
svc = get_boundry(Zero_one)

ax = plt.gca()
plt.figure(figsize=(5.5,5.5))     

axes = [0, 100, 0, 100]
x0s = np.linspace(axes[0], axes[1], 100)
x1s = np.linspace(axes[2], axes[3], 100)
x0, x1 = np.meshgrid(x0s, x1s)
X = np.c_[x1.ravel(), x0.ravel()]
y_pred = svc.predict(X)
y_pred = y_pred.reshape(x0.shape)
#plt.contourf(x0, x1, y_pred, colors=['white', 'blue', 'green','red'], cmap=plt.cm.brg
plt.contourf(x0, x1, y_pred+1, colors=['white','green','yellow','r'], alpha=0.8)  
#plt.colorbar(fraction=0.03, pad=0.05)  
#ax.invert_yaxis() 
plt.xlabel('belief_P',fontsize=16)
plt.ylabel('belief_N',fontsize=16)
plt.savefig('Map.eps', format='eps')
plt.show()



'''
for k in range(1):
    plt.figure(figsize=(5.5,5.5))     
    ax = plt.gca() 
    plt.xlabel('belief_P',fontsize=16)
    plt.ylabel('belief_N',fontsize=16)
    
    #print(np.max(Zeor_one[:,:,5]))
    #print(np.sum(Zeor_one[:,:,15]))

    colors=['white','yellow','green','r']
    cmap = mpl.colors.ListedColormap(colors)
    im = plt.imshow(Zero_one[:,:,24+k], cmap=cmap) 
    #im = plt.imshow(Result[:,:,20,2])
    #cb1 = plt.colorbar(im, fraction=0.03, pad=0.05)
    
    ax.invert_yaxis()
    tick_locator = ticker.MaxNLocator(nbins=4)
    #cb1.locator = tick_locator
    #cb1.update_ticks()  
        
    plt.savefig('Map_linear.eps', format='eps')
    plt.show()
    #input("PRESS ANY KEY TO CONTINUE.")
    #plt.close(plt)
'''   