from pca import *
from decimal import Decimal

import matplotlib.pyplot as plt

#------------------------------------------------------------------------------
### Cluster plot of principle components

# scipy.linalg.svd returns "Vh", which is the Hermitian (transpose)
# of the vector V. So, for us to obtain the correct V, we transpose:
V = Vh.T    

# Project the centered data onto principal component space
Z = Y @ V


# Cluster plots with all components against each other

sigcomp = 6
f=figure(figsize=(30,30))

for i in range(sigcomp):
    for j in range(sigcomp):
        plt.subplot(sigcomp+1, sigcomp, ((i+1)*sigcomp+(j+1)))
        for c in range(C):
            # select indices belonging to class c:
            class_mask = y==c
            plot(Z[class_mask,i], Z[class_mask,j], 'o', alpha=.5)
            legend(classNames)
        xlabel('PC{0}'.format(i+1))
        ylabel('PC{0}'.format(j+1))


# Cluster plot of specific components i and j
i = 0
j = 1

# Plot PCA of the data
f = figure()
title('NanoNose data: PCA')
#Z = array(Z)
for c in range(C):
    # select indices belonging to class c:
    class_mask = y==c
    plot(Z[class_mask,i], Z[class_mask,j], 'o', alpha=.5)
legend(classNames)
xlabel('PC{0}'.format(i+1))
ylabel('PC{0}'.format(j+1))

# Output result to screen
show()

#------------------------------------------------------------------------------
###
f=figure(figsize=(10,5))
pcs = [0,1,2]
legendStrs = ['PC'+str(e+1) for e in pcs]
c = ['r','g','b']
bw = .2
r = np.arange(1,M+1)
for i in pcs:    
    plt.bar(r+i*bw, V[:,i], width=bw)
plt.xticks(r+bw, attributeNames)
plt.xlabel('Attributes')
plt.ylabel('Component coefficients')
plt.legend(legendStrs)
plt.grid()
plt.title('Body Style:: PCA Component Coefficients')
plt.xticks(rotation = 45)
plt.show()

#------------------------------------------------------------------------------
### Plot explanation of principle components
threshold = 0.95

f=figure(figsize=(10,7))
# Plot variance explained
plt.plot(range(1,len(rho)+1),rho,'x-')

for x,y in zip(range(1,len(rho)+1),rho):
    
    label = "{:.2f}".format(y)

    plt.annotate(label, # this is the text
                 (x,y), # this is the point to label
                 textcoords="offset points", # how to position the text
                 xytext=(0,5), # distance from text to points (x,y)
                 ha='center') # horizontal alignment can be left, right or center

plt.plot(range(1,len(rho)+1),np.cumsum(rho),'o-')

for x,y in zip(range(1,len(rho)+1),np.cumsum(rho)):

    label = "{:.2f}".format(y)

    plt.annotate(label, # this is the text
                 (x,y), # this is the point to label
                 textcoords="offset points", # how to position the text
                 xytext=(0,5), # distance from text to points (x,y)
                 ha='center') # horizontal alignment can be left, right or center

plt.plot([1,len(rho)],[threshold, threshold],'k--')
plt.title('Variance explained by principal components');
plt.xlabel('Principal component');
plt.ylabel('Variance explained');
plt.legend(['Individual','Cumulative','Threshold'])
plt.grid()
plt.show()
