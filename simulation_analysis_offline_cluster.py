import numpy as np
import sys
import simulation_analysis as sa


if __name__ == '__main__':
  # Set parameters
  file_prefix = '/home/fbalboa/simulations/RigidMultiblobsWall/rheology/data/run2000/run2110/run2110.5.0.0'
  files_config = ['/home/fbalboa/simulations/RigidMultiblobsWall/rheology/data/run2000/run2110/run2110.5.0.0.star_run2110.5.0.0.config'] 
  name_vertex = '/home/fbalboa/sfw/RigidMultiblobsWall/multi_bodies/examples/rheology/Structures/star_N_13_a_0.05.vertex'
  num_frames = 10
  num_frames_skip_fraction = 4
  rcut = 0.1
  nbins = 100
  dim = '3d' # '3d', '2d' or 'q2d'
  gr_type = 'blobs' # 'bodies' or 'blobs'
  L_gr = np.array([5, 5, 50])
  
  # Read input file
  name_input = file_prefix + '.inputfile'
  read = sa.read_input(name_input)
  print('name_input  = ', name_input)  
  
  # Read vertex
  r_vectors = sa.read_vertex(name_vertex)

  # Get number of particles
  N = sa.read_particle_number(files_config[0])

  # Set some parameters
  if L_gr is None:
    L = np.fromstring(read.get('periodic_length'), sep=' ')
  else:
    L = L_gr   
  print('L         = ', L)
  print(' ')

  # Read config
  x = sa.read_config_list(files_config, print_name=True)
  num_frames = x.shape[0] if x.shape[0] < num_frames else num_frames
  if num_frames_skip_fraction > 0:
    num_frames_skip = num_frames // num_frames_skip_fraction
  else:
    num_frames_skip = 0
  print('total number of frames = ', x.shape[0])
  print('num_frames             = ', num_frames)
  print('num_frames_skip        = ', num_frames_skip)
  print(' ')

  # Skip frames
  x = x[num_frames_skip:]

  # Detect clusters
  clusters = sa.cluster_detection(x, num_frames, rcut=rcut, r_vectors=r_vectors, L=L)

  # Fraction of bodies in a cluster
  sel = clusters.flatten() > -1
  fraction_in_cluster = np.sum(sel) / sel.size
  print('fraction_in_cluster = ', fraction_in_cluster)

  # Cluster size histogram
  num_colloids_in_cluster = []
  for j, cj in enumerate(clusters):
    for k in range(cj.size):
      index = cj[k]
      if index == k:
        colloids_indexes = np.argwhere(cj == index)
        num_colloids_in_cluster.append(colloids_indexes.size)
      if index == -1:
        num_colloids_in_cluster.append(1)
  num_colloids_in_cluster = np.array(num_colloids_in_cluster, dtype=int).flatten() 
  name = file_prefix + '.histogram.cluster_size.dat'
  sa.compute_histogram(num_colloids_in_cluster, num_intervales=np.max(num_colloids_in_cluster)+1, xmin=-0.5, xmax=np.max(num_colloids_in_cluster)+0.5,  name=name)

  # Cluster mean size
  Nc = np.sum(num_colloids_in_cluster) / num_colloids_in_cluster.size
  print('Nc = ', Nc)  
  sel = num_colloids_in_cluster > 1
  Nc = np.sum(num_colloids_in_cluster[sel]) / num_colloids_in_cluster[sel].size
  print('Nc = ', Nc)

  # Cluster length

  # Fractal geometry  

  # Escape time 

  # xyz 

  
  

  
  
