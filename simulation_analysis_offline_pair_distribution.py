'''
Average velocity profiles and use them to extract the viscosity.
'''
import numpy as np
import sys
import simulation_analysis as sa


if __name__ == '__main__':
  file_prefix = '/home/fbalboa/simulations/RigidMultiblobsWall/sedimentation/data/dipankar/run5/run_sedimentation'
  files_config = ['/home/fbalboa/simulations/RigidMultiblobsWall/sedimentation/data/dipankar/run5/run_sedimentation.sphere_colloids.config']
  files_config_B = ['/home/fbalboa/simulations/RigidMultiblobsWall/sedimentation/data/dipankar/run5/run_sedimentation.cube_colloids.config']
  reference_particles = 100 + np.arange(50, dtype=int) # or None
  name_vertex = None
  skiprows_vertex = 0
  simulation_number_start = 0
  simulation_number_end = 25
  N_skip_fraction = 1000000
  rcut = 3.5
  nbinsx = 20
  nbinsy = 1
  nbinsz = 40
  dim = '2d_radial'
  Lz_wall = np.array([-np.inf, np.inf])  
  suffix_output = '.cube_all.pair_distribution.vtk'

  # Set output name
  output_name = file_prefix + suffix_output
  print('output_name = ', output_name)

  # Read inputfile
  name_input = file_prefix + '.inputfile' 
  read = sa.read_input(name_input)

  # Read vertex file
  if name_vertex:
    r_vectors = np.loadtxt(name_vertex, skiprows=skiprows_vertex)
  else:
    r_vectors = None
    
  # Set some parameters
  L = np.fromstring(read.get('periodic_length') or '0 0 0', sep=' ')
  print('L   = ', L)
  print(' ')

  # Get files
  if len(files_config) is None:
    files_config = []
    for i in range(simulation_number_start, simulation_number_end + 1):
      name = file_prefix + str(i) + suffix + str(i) + '.config'
      files_config.append(name)

  # Loop over config files
  x = sa.read_config_list(files_config, print_name=True)

  if files_config_B:
    x_B = sa.read_config_list(files_config_B, print_name=True)
    N_A = x.shape[1]
    N_B = x_B.shape[1]
    x_new = np.zeros((x.shape[0], N_A + N_B, x.shape[2]))
    x_new[:,0:N_A,:] = x
    x_new[:,N_A:,:] = x_B
    x = x_new

  
  # Get number of frames and skip frames
  num_frames = x.shape[0]
  N_skip = int(num_frames // N_skip_fraction)
  print('num_frames = ', num_frames)
  print('N_skip     = ', N_skip)
  
 
  _ = sa.pair_distribution_function(x[N_skip:],
                                    num_frames - N_skip,
                                    rcut=rcut,
                                    nbinsx=nbinsx,
                                    nbinsy=nbinsy,
                                    nbinsz=nbinsz,
                                    r_vectors=r_vectors,
                                    L=L,
                                    Lz_wall=Lz_wall,
                                    offset_walls=True,
                                    dim=dim,
                                    name=output_name,
                                    reference_particles=reference_particles)

