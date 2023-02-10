import numpy as np
import imageio
from matplotlib import pyplot as plt
import sys
import os

def mykmeans(pixels, K):

    # Step 1: Initialize starting cluster centers

    pixels = pixels.astype(np.float32)
    pixels = pixels.reshape(pixels.shape[0]*pixels.shape[1],pixels.shape[2])
    N,D = pixels.shape

    c_indx = np.random.choice(np.arange(N),[K,],replace=False) # Choosing K points/ cells from total of N cells (rows) of data

    # The K cells (rows) are actually made the random cluster centers.
    #c_indx = np.array([1,2,3,4,5]) # intentional

    old_c = pixels[c_indx] # random initialisatied center. (K,3)
    old_clusters = np.zeros([N]) # N = Pixel shape. The cluster association is for each cell (row) values from (0-K-1)

    # Step 2: Iterate through K-Means

    M = old_c.shape[0] # Number of clusters (K). old_c = (K,3)
    pix = np.reshape(pixels,[N,1,D]) # reshape the pixels into L*W, 1, 3.
    centroid = np.zeros_like(old_c) # Kx3 , K centroids will be there which will be further updated.
    max_iter = 100 # Limit of iterations for convergence.
    itr = 0 # Iteration value at the start of computation.

    while True:

        # Step Computing pairwise distances between each data point and the cluster centers        
        cen = np.reshape(old_c,[1,M,D]) # 1,K,3 to make it conform to pix , pix is L*W, 1, 3.
        dist = np.sqrt(np.sum((pix-cen)**2,axis=2)) # Compute distance of each cell (row) from the three clusters. L*W , K for each pixel (row) compute distance from all the K centers (pixels)

        # Step Assign data points to clusters
        # from the K cols of the dist at each row (pixel), find the minimum column value and assign it's index to clusters.
        clusters = np.argmin(dist,axis=1) # (0-K) minimum index assignment.

        # Step Update centers

        # Iterate from 0-K , i.e pick each pixel with a particular cluster value and perform the mean operation
        for i in range(K):
            # Get indices of each point in cluster i
            indxs = np.argwhere(clusters==i).flatten()
            # Collect the points of cluster i
            data = pixels[indxs]
            # Calculate mean of points in cluster i
            new_mean = np.mean(data,axis=0) # Row wise mean value of the 3 columns.
            # Add new centers
            centroid[i] = new_mean # The updated color (centroid) value of all the K cluster centers.

        # Step Check if centers have changed (convergence)

        # check if old clusters are the same as new ones , clusters are (K,3)
        if(np.array_equal(old_clusters,clusters)):
            break
        elif(itr==max_iter): # If iter value reached
            break
        else:
            itr += 1
            old_c = np.copy(centroid)
            old_clusters = np.copy(clusters)

    # return the cluster assignment to each pixel (row) and return the color (R,B,G) of each assumed cluster center. 
    print(itr)       
    return clusters, centroid

def mykmedoids(pixels, K):


    # Initial formulation same as K-means
    pixels = pixels.astype(np.float32)
    pixels = pixels.reshape(pixels.shape[0]*pixels.shape[1],pixels.shape[2])
    N,D = pixels.shape
    max_iter = 500 
    itr = 0

    # Select k random points out of the data points to start at medoids  
    m_ind = np.random.choice(np.arange(N),[K,],replace=False)    
    #m_ind = np.array([1,2,3,4,5]) # intentional
    medoids = pixels[m_ind] 

    
    while itr <= max_iter:
        # Associate each data point to closest medoid by Euclidean distance
        # Compute the distance of each medoid by the distance method.
        dist = com_d(pixels,medoids) 
        # (0-K) minimum index assignment.
        labels = np.argmin(dist,axis=1) # Similar to k-means associate the labels of each pixel according to the minium index value medoid.
    
        # Update the medoids
        old_medoids = medoids.copy()

        #for i in range(K):
        for i in set(labels): # labels represent each pixel LxB
            diss_av = np.sum(com_d(pixels,np.reshape(medoids[i],(1,D))))
            clusters = pixels[labels==i]
        
            # iterate through all the points in the cluster and find the best possible object
        for point in clusters:
            new_medoid = point
            new_dissimilarity = np.sum(com_d(pixels,np.reshape(point,(1,D))))
            
            if new_dissimilarity < diss_av:
                diss_av = new_dissimilarity
                
                medoids[i] = point
        
        itr += 1

        # check if old medoids are the same as new ones , medoids.
        if(np.array_equal(old_medoids,medoids)):
            break
        elif(itr==max_iter):
            break

    return labels, medoids
    

def com_d(X,medoids):
    N,D = X.shape
    K = medoids.shape[0]
    return np.sqrt(np.sum((np.reshape(X,[N,1,D])-np.reshape(medoids,[1,K,D]))**2,axis=2))        

def main():
	if(len(sys.argv) < 2):
		print("Please supply an image file")
		return

	image_file_name = sys.argv[1]
	K = 5 if len(sys.argv) == 2 else int(sys.argv[2])
	im = np.asarray(imageio.imread(image_file_name))

	fig, axs = plt.subplots(1, 2)

	classes, centers = mykmedoids(im, K)

	new_im = np.asarray(centers[classes].reshape(im.shape), im.dtype)
	imageio.imwrite(os.path.basename(os.path.splitext(image_file_name)[0]) + '_converted_mykmedoids_' + str(K) + os.path.splitext(image_file_name)[1], new_im)
	axs[0].imshow(new_im)
	axs[0].set_title('K-medoids')

	classes, centers = mykmeans(im, K)
    
	new_im = np.asarray(centers[classes].reshape(im.shape), im.dtype)
	imageio.imwrite(os.path.basename(os.path.splitext(image_file_name)[0]) + '_converted_mykmeans_' + str(K) + os.path.splitext(image_file_name)[1], new_im)
	axs[1].imshow(new_im)
	axs[1].set_title('K-means')

	plt.show()

if __name__ == '__main__':
	main()