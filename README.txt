This program gets a wav file and an initial centroids file. It calculates new centroids using the Kmeans algorithm
for up to 30 iterations( or convergence) and outputs three files:
	output_file_name - the new centroids after each iteration
	sample.compressed.wav - a compressed version of the wav file
	loss_graph.png - the average loss plot over the iterations

to run the code:
kmeans.py wav_file_name intial_centroids output_file_name

axample:
kmeans.py sample.wav cents1.txt output1.txt