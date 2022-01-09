import numpy as np
import sys
import simulation_analysis as sa


if __name__ == '__main__':
  # Set parameters
  file_prefix = '/home/fbalboa/simulations/RigidMultiblobsWall/rheology/data/run2000/run2110/run2110.2.0.0'
  files_config = ['/home/fbalboa/simulations/RigidMultiblobsWall/rheology/data/run2000/run2110/run2110.2.0.0.star_run2110.2.0.0.config'] 
  name_vertex = '/home/fbalboa/sfw/RigidMultiblobsWall/multi_bodies/examples/rheology/Structures/star_N_13_a_0.05.vertex'
  num_frames = 200
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
  
  # Loop over config files
  x = []
  for j, name_config in enumerate(files_config):
    print('name_config = ', name_config)
    xj = sa.read_config(name_config)
    if j == 0 and xj.size > 0:
      x.append(xj)
    elif xj.size > 0:
      x.append(xj[1:])

  # Concatenate config files
  x = np.concatenate([xi for xi in x])
  num_frames = x.shape[0] if x.shape[0] < num_frames else num_frames
  if num_frames_skip_fraction > 0:
    num_frames_skip = num_frames // num_frames_skip_fraction
  else:
    num_frames_skip = 0
  print('total number of frames = ', x.shape[0])
  print('num_frames             = ', num_frames)
  print('num_frames_skip        = ', num_frames_skip)
  print(' ')

  # Detect clusters
  sa.timer('cluster')
  clusters = sa.cluster_detection(x, num_frames, rcut=rcut, r_vectors=r_vectors, L=L)
  sa.timer('cluster')

  print('clusters = ', clusters.shape)

  sa.timer('', print_all=True)

  
