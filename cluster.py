from sklearn.cluster import DBSCAN
from sklearn.preprocessing import normalize
from sklearn import metrics
from numpy.linalg import norm
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

file = open('./data/um_buildings_watts_aggregates_with_names.csv', 'r')
l1 = file.readline()
names = l1.split(',')
print names
data = []
for i in range(0, len(names)):
    data.append([])

print "Reading data..."
for line in file:
    spl = line.split(',')
    for i in range(0, len(names)):
        data[i].append(float(spl[i]))
    
print "\n\n\n\n\n"
print "-------------------DATA COLLECTED-------------------"
print "Creating profiles..."
profiles = []
n_samples = 720

for building in data:
    profile = [0]*n_samples
    i=0
    for i in range(0, len(building)):
       profile[i%n_samples] += building[i]
        
    for j in range(0, len(profile)):
        if (j<(i%n_samples)):
            profile[j] = profile[j]/(i/n_samples + 1)
        else:
            profile[j] = profile[j]/(i/n_samples)
    profiles.append(normalize(np.array(profile).reshape(1,-1), norm='l2')[0])
    #profiles.append(profile)


    

"""
===================================
Demo of DBSCAN clustering algorithm
===================================

Finds core samples of high density and expands clusters from them.

"""

X = profiles
##############################################################################
# Compute DBSCAN

print "Generating clusters..."
db = DBSCAN(eps=0.025, min_samples=3).fit(X)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_
print "\n\n\n"
print len(labels)
print len(names)
print len(db.components_)
print labels

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
size_clusters = [0]*(n_clusters_ + 1)
print('\nEstimated number of clusters: %d' % n_clusters_)

##############################################################################
# Plot result
import matplotlib.pyplot as plt

for i in range(len(labels)):
    plt.figure(labels[i]+1)
    size_clusters[labels[i]+1] += 1
    plt.plot(X[i])
    plt.axis([0,720,0,0.08])

for i in range(len(size_clusters)):
    print "%scluster %d: %d features" % ("noisy " if i==0 else "", i, size_clusters[i])
    
plt.show()
#
# Black removed and is used for noise instead.
#unique_labels = set(labels)
#colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
#for k, col in zip(unique_labels, colors):
#    if k == -1:
#        # Black used for noise.
#        col = 'k'
#
#    class_member_mask = (labels == k)
#
#    xy = X[class_member_mask & core_samples_mask]
#    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col,
#             markeredgecolor='k', markersize=14)
#
#    xy = X[class_member_mask & ~core_samples_mask]
#    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col,
#             markeredgecolor='k', markersize=6)
#
#plt.title('Estimated number of clusters: %d' % n_clusters_)
#plt.show()
 
    