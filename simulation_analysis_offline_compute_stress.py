'''
Script to compute the stress from the blob_forces.
'''
import numpy as np
import sys
import simulation_analysis as sa
from quaternion import Quaternion

if __name__ == '__main__':
  # Set parameters
  file_prefix = '/home/fbalboa/simulations/RigidMultiblobsWall/rheology/data/run3000/run3005/run3005.1.0.0'
  files_method = 'File' # 'sequence'
  file_start = 0
  file_end = 0
  file_prefix_config = None
  file_suffix_config = None
  files_config = ['/home/fbalboa/simulations/RigidMultiblobsWall/rheology/data/run3000/run3005/run3005.1.0.0.shell_run3005.1.0.0.config']
  files_blob_forces = ['/home/fbalboa/simulations/RigidMultiblobsWall/rheology/data/run3000/run3005/run3005.1.0.0.blob_forces.dat']
  name_vertex = '/home/fbalboa/sfw/RigidMultiblobsWall/multi_bodies/Structures/shell_N_162_Rg_0_9497_Rh_1.vertex'
  output_name = '/home/fbalboa/simulations/RigidMultiblobsWall/rheology/data/run3000/run3005/run3005.1.0.0.postprocessing.stress.dat'
  num_frames = 10000
  n_save_xyz = 1
  mode = 'no_projection'
  save_dat_index = []

  # Get names config files
  if files_method == 'sequence':
    files_config = []
    for i in range(file_start, file_end + 1):
      name = file_prefix_config + str(i) + file_suffix_config + str(i) + '.config'
      files_config.append(name)
  
  # Read inputfile
  name_input = file_prefix + '.inputfile' 
  read = sa.read_input(name_input)

  # Read vertex
  r_vectors = sa.read_vertex(name_vertex)

  # Get number of particles
  N = sa.read_particle_number(files_config[0])

  # Set some parameters
  dt = float(read.get('dt')) 
  n_save = int(read.get('n_save'))
  dt_sample = dt * n_save * n_save_xyz
  eta = float(read.get('eta'))
  periodic_length = np.fromstring(read.get('periodic_length'), sep=' ')
  print('file_prefix     = ', file_prefix)
  print('dt              = ', dt)
  print('dt_sample       = ', dt_sample)
  print('n_save          = ', n_save)
  print('eta             = ', eta)
  print('periodic_length = ', periodic_length)
  print(' ')

  # Loop over config files
  x = sa.read_config_list(files_config, print_name=True)
  x = x[0:-1]

  # Read force on blobs
  blob_forces = sa.read_config_list(files_blob_forces, print_name=True)
  blob_forces = blob_forces[0::n_save_xyz]

  # Get time
  t = np.arange(x.shape[0]) * dt_sample
  print('total number of frames = ', x.shape[0])
  print('num_frames             = ', num_frames)
  print(' ')


  print('x = ', x.shape)
  print('blob_forces = ', blob_forces.shape)
  # sys.exit()

  # Compute stress
  sa.compute_stress(x, r_vectors, blob_forces, output_name, periodic_length=periodic_length, save_dat_index=save_dat_index, mode=mode)

        
