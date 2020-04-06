import sys
import scipy.io.wavfile as wf
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt

ITER = 30

def run_kmeans(sample,centroids,file):
    #read the sample data
    fs, y = wf.read(sample)
    
    train_set = np.array(y.copy())
    centroids = np.loadtxt(centroids)
    centroids_old =  np.zeros(centroids.shape) 
    distances = np.zeros((len(train_set),len(centroids)))
    clusters = np.zeros(train_set.shape)
    epoch = 0
    converge = False
    graph_data = []
    compressed = []
    
    while epoch < ITER and not converge:
        epoch += 1
        #calculate the distance of each train point to each center
        distances = np.swapaxes(np.sqrt(((train_set - centroids[:, np.newaxis])**2).sum(axis=2)),0,1)
        #assign each train point to its nearest center
        clusters = np.argmin(distances, axis=1)
        
        #calc new centers
        centroids_old = deepcopy(centroids)
        for i in range(len(centroids)):
            centroids[i] = np.round(np.mean(train_set[clusters==i], axis=0))
        
        converge = np.linalg.norm(centroids - centroids_old) == 0
        #print(f"[iter {epoch-1}]:{','.join([str(i) for i in centroids])}")
        file.write(f"[iter {epoch-1}]:{','.join([str(i) for i in centroids])}\n")
        
        #calc loss
        graph_data.append(np.mean(np.min(distances,axis=1)))
    plot_res(epoch,graph_data)
    
    for i in clusters:
        compressed.append(centroids_old[i])
    wf.write("sample.compressed.wav",fs,np.array(compressed,np.int16))
    
def plot_res(epoch,graph_data):
    plt.plot(graph_data)
    plt.ylabel('Average loss')
    plt.xlabel('Epoch')
    plt.xticks(np.arange(epoch))
    plt.savefig("loss_graph.png")
if __name__ == '__main__':
    sample, centroids,output = sys.argv[1], sys.argv[2],sys.argv[3
                                       ]
    #sample = "sample.wav"
    #centroids = "cents2.txt"
    #output = "output2.txt"
    
    file = open(output,"w")
    run_kmeans(sample,centroids,file)
    file.close()
