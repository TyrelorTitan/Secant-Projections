# -*- coding: utf-8 -*-
"""
Created on Wed May  7 11:13:27 2025

@author: agilj

This is a script that implements the PCA Secant projections described in:
    
Li, Y. et al., "Compressive image acquisition and classification via secant
projections," Journal of Optics 17.6, 2015.


"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import svd
from scipy.sparse.linalg import svds as partialSVD
from math import comb as nchoosek
from sklearn import svm

"""
Class to manage the training of secant projections.
"""
class secants():
    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)
    
    def __call__(self, data, numSecants, numSamples=10000):
        return self._trainSecants_PCA(data, numSecants, numSamples)
    
    def _trainSecants_PCA(self, data, numSecants, numSamples):
        numClasses = len(data)
        sec_inter = []
        # Class-based or class-free secants?
        if numClasses>1: # We have class info.
            # Sample the secant manifold.
            for ii in range(numClasses-1):
                for jj in range(ii+1,numClasses):
                    # If numSamples is set to be greater than the number of
                    # of pairwise secants we can compute, then use all
                    # pairwise secants we have. Otherwise, sample randomly.
                    numEx_class1 = data[ii].shape[0]
                    numEx_class2 = data[jj].shape[0]
                    if numSamples > numEx_class1 * numEx_class2:
                        # Get all unique pairs.
                        idx1, idx2 = np.meshgrid(np.arange(numEx_class1),
                                                 np.arange(numEx_class2))
                        idx1 = np.ravel(idx1)
                        idx2 = np.ravel(idx2)
                    else:
                        # Randomly sample the secant manifold.
                        idx = self.rng.permutation(numEx_class1*numEx_class2)
                        idx = idx[:numSamples]
                        
                        idx1 = np.mod(idx,numEx_class1).astype(int)
                        idx2 = (idx/numEx_class1).astype(int)
                    
                    # Get total number of pairs we are using.
                    numPairs = len(idx1) 
                    
                    # Compute secants and store them for later.
                    for kk in range(numPairs):
                        secant = data[ii][idx1[kk],:] - data[jj][idx2[kk],:]
                        sec_inter.append(secant)
                    
            # Make sure there are no 0-vectors in the secant dataset.
            sec_inter = np.array(sec_inter)
            sec_inter = sec_inter[np.any(sec_inter,axis=1)]
            
            # We now have a list of secants. Compute SVD to get a
            # projection matrix that maps to their induced
            # coordinate system.
            if numSecants < np.min(sec_inter.shape):
                U,S,_ = partialSVD(sec_inter.transpose(), k=numSecants)
            else:
                U,S,_ = svd(sec_inter.transpose())
            S = S[::-1]
            U = U[:,::-1]
            r = numSecants # Rank of the resulting projection matrix.
            return U, S, r
        
        else: # No class info.
            # Sample the secant manifold.
            numEx = data[0].shape[0]
            numCombos = nchoosek(numEx,2)
            if numSamples > numCombos:
                # Get all unique pairs.
                idx1, idx2 = np.meshgrid(np.arange(numEx),
                                         np.arange(numEx))
                idx1 = np.ravel(idx1)
                idx2 = np.ravel(idx2)
            else:
                # Randomly sample the secant manifold.
                idx = self.rng.permutation(numEx*numEx)
                idx = idx[:numSamples]

                idx1 = np.mod(idx,numEx).astype(int)
                idx2 = (idx/numEx).astype(int)
                
            # Now compute secants.
            numPairs = len(idx1)
            for kk in range(numPairs):
                secant = data[0][idx1[kk],:] - data[0][idx2[kk],:]
                sec_inter.append(secant)

            # Make sure there are no 0-vectors in the secant dataset.
            sec_inter = np.array(sec_inter)
            sec_inter = sec_inter[np.any(sec_inter,axis=1)]
            
            # We now have a list of secants. Compute SVD to get a
            # projection matrix that maps to their induced
            # coordinate system.
            if numSecants < np.min(sec_inter.shape):
                U,S,_ = partialSVD(sec_inter.transpose(), k=numSecants)
            else:
                U,S,_ = svd(sec_inter.transpose())
            S = S[::-1]
            U = U[:,::-1]
            r = numSecants # Rank of the resulting projection matrix.
            return U, S, r
    
"""
Accepts a list X, where each element in the list is a different class. Each
list should be N_i x F where N_i is the number of examples in that class and F
is the number of features.
Returns a tuple (Xout,y) where Xout is an NxF ndarray, where N is the total
number of examples in the list, and where y is a length-N array of class
labels.
"""
def format_SVM_data(X):
    y = []
    Xout = []
    for idx, cl in enumerate(X):
        label = idx * np.ones((cl.shape[0],))
        y.extend(label)
        Xout.extend(cl)
    Xout = np.array(Xout)
    y = np.array(y)
    return (Xout,y)

#%% Code demonstration using a simple 1-D example.
if __name__ == '__main__':
    """ Code Demo """
    # Example data.
    rng = np.random.default_rng()
    numFeats = 101
    
    # Class parameters (Gaussians as an example)
    height_c1 = 1
    FWHM_c1 = 0.3
    sigma_c1 = FWHM_c1 / (2*np.sqrt(2*np.log(2)))
    numEx_c1 = 100
    
    height_c2 = 1
    FWHM_c2 = 0.3
    sigma_c2 = FWHM_c2 / (2*np.sqrt(2*np.log(2)))
    numEx_c2 = 100
    
    # Construct class Stereotypes
    x = np.linspace(0,1,numFeats)
    class1 = height_c1*np.exp(-(x-0.2)**2 / (2*sigma_c1**2))
    class2 = height_c2*np.exp(-(x-0.3)**2 / (2*sigma_c2**2))

    plt.figure(dpi=600)
    plt.plot(x,class1)
    plt.plot(x,class2)
    plt.legend(['Class 1', 'Class 2'])
    plt.title('Example Class Stereotypes')
    plt.show()
    
    #%% Train secant projections and an SVM for classification.
    # Measurement noise.
        # Class 1
    noiseSigma_c1_train = 0.1
    noiseBias_c1_train = 0
    noise_c1 = rng.normal(noiseBias_c1_train,noiseSigma_c1_train,
                          size=(numEx_c1,numFeats))

        # Class 2
    noiseSigma_c2_train = 0.1
    noiseBias_c2_train = 0
    noise_c2 = rng.normal(noiseBias_c2_train,noiseSigma_c2_train,
                          size=(numEx_c2,numFeats))

    # Now build 'measured' dataset as 'true' value plus noise.
    c1 = class1 + noise_c1
    c2 = class2 + noise_c2
    
    # Display example training data.
    plt.figure(dpi=600)
    plt.plot(x,c1[0,:])
    plt.plot(x,c2[0,:])
    plt.legend(['Class 1', 'Class 2'])
    plt.title('Example Training Data')
    plt.show()
    
    # Make secant generator.
    numSecants = 2 # Really, we only need "numClasses-1" from a 
                   # degrees-of-freedom argument.
    secantLearner = secants()
    
    # Data format should be list of np.ndarrays. Each element in the list is
    # a class. Each row in the np.ndarray is an example. Each column in the
    # np.ndarray is a feature. All np.ndarrays should have the same number
    # of columns.
        # Noisy Training Data
    data_forSecants = [c1, c2]
        # Domain Knowledge of what classes look like.
    # data_forSecants = [np.tile(class1,(numSecants,1)),
    #                    np.tile(class2,(numSecants,1))]
    
    # Train secants.
    sec, strength, rank = secantLearner(data_forSecants, 
                                        numSecants, 
                                        numSamples=9999)
    
    # Display learned projections.
    legend = []
    plt.figure(dpi=600)
    for i in range(2):
        plt.plot(x,sec[:,i] * np.sqrt(strength[i]))
        legend.append('Projection '+str(i))
    plt.title('First '+str(i+1)+' Projections')
    plt.legend(legend)
    
    # Now project data onto secant vectors.
    secData_c1 = c1 @ sec * np.sqrt(strength)
    secData_c2 = c2 @ sec * np.sqrt(strength)
    
    # Visualize separation.
    plt.figure(dpi=600)
    plt.plot(secData_c1[:,0],secData_c1[:,1],'b.')
    plt.plot(secData_c2[:,0],secData_c2[:,1],'r.')
    plt.title('Training Data Separation')
    plt.xlabel('Secant 1')
    plt.ylabel('Secant 2')
    
    # Build SVM-compatible dataset
    data_forSVM, labels_forSVM = format_SVM_data([secData_c1,secData_c2])

    # Train an SVM
    svm_classifier = svm.SVC(kernel='linear',gamma='scale')
    svm_classifier.fit(data_forSVM, labels_forSVM)
    
    #%% Check performance on a test dataset.
    # Make the test set
    numTest_c1 = 100
    numTest_c2 = 100
        # Class 1
    noiseSigma_c1_test = 1
    noiseBias_c1_test = 0
    noise_c1_test = rng.normal(noiseBias_c1_test,noiseSigma_c1_test,
                          size=(numTest_c1,numFeats))
        # Class 2
    noiseSigma_c2_test = 1
    noiseBias_c2_test = 0
    noise_c2_test = rng.normal(noiseBias_c2_test,noiseSigma_c2_test,
                          size=(numTest_c2,numFeats))
        # Now build 'measured' dataset as 'true' value plus noise.
    c1_test = class1 + noise_c1_test
    c2_test = class2 + noise_c2_test
    
    # Plot example test data.
    plt.figure(dpi=600)
    plt.plot(x,c1_test[0,:])
    plt.plot(x,c2_test[0,:])
    plt.legend(['Class 1', 'Class 2'])
    plt.title('Example Test Data')
    plt.show()

    # Compute secant projections
    secData_c1_test = c1_test @ sec * np.sqrt(strength)
    secData_c2_test = c2_test @ sec * np.sqrt(strength)
    
    # Format data to go into SVM
    data_test, labels_test = format_SVM_data([secData_c1_test,secData_c2_test])
    
    # Test SVM
    preds = svm_classifier.predict(data_test)
    
    # Check accuracy
    gotCorrect = preds==labels_test
    acc = np.mean(gotCorrect)*100
    print('Accuracy: '+str(acc)+'%.')
    
    # Make cute visualization of decision boundary
        # Get min and max x-points (since this is a 1-D problem)
    minx = np.minimum(secData_c1_test[:,0].min(),secData_c2_test[:,0].min())
    maxx = np.maximum(secData_c1_test[:,0].max(),secData_c2_test[:,0].max())
    minx = minx*1.1
    maxx = maxx*1.1
        # Get grid points to visualize decision boundary
    numVisPts = 100
    ax1 = np.linspace(minx,maxx,numVisPts)
    # ax2 = np.linspace(-1,1,numVisPts) # For Domain Knowledge plot
    ax2 = np.linspace(-15,15,numVisPts) # For training data plot
    ax1,ax2 = np.meshgrid(ax1,ax2)
    dataVis = np.concatenate((ax1[:,:,None],ax2[:,:,None]),axis=2)
    dataVis = np.reshape(dataVis,(numVisPts**2,2),)
        # Compute SVM label for each grid point.
    cuteVis = svm_classifier.predict(dataVis)
        # Display output.
    plt.figure(dpi=600)
    plt.plot(dataVis[cuteVis==0,0],dataVis[cuteVis==0,1],'g.',markersize=5)
    plt.plot(dataVis[cuteVis==1,0],dataVis[cuteVis==1,1],'c.',markersize=5)
    plt.plot(secData_c1_test[:,0],secData_c1_test[:,1],'r.')
    plt.plot(secData_c2_test[:,0],secData_c2_test[:,1],'b.')
    plt.title('Decision Boundary')
    # plt.text(1.3*np.mean([minx,maxx]),0.5,'Acc: '+str(acc)) # Domain Knowledge
    plt.text(1.3*np.mean([minx,maxx]),12,'Acc: '+str(acc)) # Trainng data

