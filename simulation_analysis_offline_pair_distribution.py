'''
Average velocity profiles and use them to extract the viscosity.
'''
import numpy as np
import sys
import simulation_analysis as sa


if __name__ == '__main__':
  file_prefix = '/home/fbalboa/simulations/RigidMultiblobsWall/rheology/data/run2000/run2123/run2123.5.0.'
  suffix = '.star_run2123.5.0.'
  name_vertex = '/home/fbalboa/simulations/RigidMultiblobsWall/rheology/data/star_hook_N_33_a_0.05.vertex'
  # name_vertex = None
  skiprows_vertex = 10
  simulation_number_start = 0
  simulation_number_end = 25
  N_skip_fraction = 1.1
  rcut = 0.3
  nbins = 80
  Lz_wall = np.array([0, 4.5])
  suffix_output = 'base.pair_distribution_blobs.nbins.' + str(nbins) + '.vtk'

  # Set output name
  output_name = file_prefix + suffix_output
  print('output_name = ', output_name)

  # Read inputfile
  name_input = file_prefix + '0.inputfile' 
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
  files_config = []
  for i in range(simulation_number_start, simulation_number_end + 1):
    name = file_prefix + str(i) + suffix + str(i) + '.config'
    files_config.append(name)

  # Loop over config files
  x = sa.read_config_list(files_config, print_name=True)

  # Get number of frames and skip frames
  num_frames = x.shape[0]
  N_skip = int(num_frames // N_skip_fraction)
  print('num_frames = ', num_frames)
  print('N_skip     = ', N_skip)
  
 
  _ = sa.pair_distribution_function(x[N_skip:], num_frames - N_skip, rcut=rcut, nbins=nbins, r_vectors=r_vectors, L=L, Lz_wall=Lz_wall,
                                    offset_walls=True, dim='3d', name=output_name)
  
