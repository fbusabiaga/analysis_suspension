import numpy as np
import sys
import simulation_analysis as sa


if __name__ == '__main__':
  # Set parameters
  file_prefix = '/workspace/scratch/users/fbalboa/simulations/RigidMultiblobsWall/rheology/data/run2000/run2123/run2123'
  structure_prefix = 'star_run'
  first_index = 6
  second_index = 0
  number_simulation = 2123
  N_samples_start = 0
  N_samples = 15
  
  # Get number of particles
  name_config = file_prefix + '.' + str(first_index) + '.' + str(second_index) + '.0.' + structure_prefix + str(number_simulation) + '.' + str(first_index) + '.' + str(second_index) + '.0.config'
  N = sa.read_particle_number(name_config)


  if False:
    # Read config
    names_config = []
    for j in range(N_samples_start, N_samples):
      names_config.append(file_prefix + '.' + str(first_index) + '.' + str(second_index) + '.' + str(j) + '.' + structure_prefix + str(number_simulation) + '.' + str(first_index) + '.' + str(second_index) + '.' + str(j) + '.config')
    x = sa.read_config_list(names_config, print_name=True)
    num_frames = x.shape[0]
    print('x.shape    = ', x.shape)
    print('num_frames = ', num_frames)
    print('N          = ', N)
    print(' ')

  else:
    # Read config
    offset = 0
    for j in range(N_samples_start, N_samples):
      names_config = [file_prefix + '.' + str(first_index) + '.' + str(second_index) + '.' + str(j) + '.' + structure_prefix + str(number_simulation) + '.' + str(first_index) + '.' + str(second_index) + '.' + str(j) + '.config']
      x = sa.read_config_list(names_config, print_name=True)
      num_frames = x.shape[0]
      print('offset    = ', offset)
      print(' ')
      offset += num_frames - 1
