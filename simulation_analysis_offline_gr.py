import numpy as np
import sys
import simulation_analysis as sa


if __name__ == '__main__':
  # Set parameters
  file_prefix = '/home/fbalboa/simulations/RigidMultiblobsWall/FreeSlip/data/run0/run60/run60.13.0.0'
  files_config = ['/home/fbalboa/simulations/RigidMultiblobsWall/FreeSlip/data/run0/run60/run60.13.0.0.particles.config']
  files_config_B = None # ['/home/fbalboa/simulations/RigidMultiblobsWall/sedimentation/data/dipankar/run5/run_sedimentation.cube_colloids.config']
  name_vertex = None
  num_frames = 10000
  num_frames_skip_fraction = 0
  rcut = 5.1
  nbins = 100
  dim = '3d' # '3d', '2d' or 'q2d'
  gr_type = 'bodies' # 'bodies' or 'blobs'
  L_gr = np.array([14.5, 14.5, 14.5]) 
  
  # Read vertex
  if name_vertex:
    r_vectors = sa.read_vertex(name_vertex)
  else:
    r_vectors = None

  # Get number of particles
  N = sa.read_particle_number(files_config[0])

  # Set some parameters
  if L_gr is None:
    # Read input file
    name_input = file_prefix + '.inputfile'
    read = sa.read_input(name_input)
    L = np.fromstring(read.get('periodic_length'), sep=' ')
  else:
    L = L_gr   
  print('L         = ', L)
  print(' ')

  # Read config
  x = sa.read_config_list(files_config, print_name=True)

  if files_config_B:
    x_B = sa.read_config_list(files_config_B, print_name=True)
    N_A = x.shape[1]
    N_B = x_B.shape[1]
    x_new = np.zeros((x.shape[0], N_A + N_B, x.shape[2]))
    x_new[:,0:N_A,:] = x
    x_new[:,N_A:,:] = x_B
    x = x_new

  # Select frames
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

