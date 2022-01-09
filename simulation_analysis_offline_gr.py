import numpy as np
import sys
import simulation_analysis as sa


if __name__ == '__main__':
  # Set parameters
  file_prefix = '/workspace/scratch/users/fbalboa/simulations/RigidMultiblobsWall/rheology/data/run2000/run2123/run2123.2.0.0'
  files_config = ['/workspace/scratch/users/fbalboa/simulations/RigidMultiblobsWall/rheology/data/run2000/run2123/run2123.2.0.0.star_run2123.2.0.0.config'] 
  name_vertex = '/workspace/scratch/users/fbalboa/sfw/RigidMultiblobsWall/multi_bodies/examples/rheology/Structures/star_hook_N_33_a_0.05.vertex'
  num_frames = 2000
  num_frames_skip_fraction = 4
  rcut = 0.3
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

  # Call gr
  name = file_prefix + '.gr.' + gr_type + '.dat'
  if gr_type == 'bodies':
    r_vectors_gr = None
  elif gr_type == 'blobs':
    r_vectors_gr = r_vectors
  gr = sa.radial_distribution_function(x[num_frames_skip:], num_frames, rcut=rcut, nbins=nbins, r_vectors=r_vectors_gr, L=L, dim=dim, name=name)

