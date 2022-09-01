import numpy as np
import sys
import simulation_analysis as sa


if __name__ == '__main__':
  # Set parameters
  file_prefix = '/workspace/scratch/users/fbalboa/simulations/RigidMultiblobsWall/rheology/data/run2000/run2113/run2113'
  structure_prefix = 'star_run'
  first_index = 5
  second_index = 0
  number_simulation = 2113
  N_samples = 20
  
  # Get number of particles
  name_config = file_prefix + '.' + str(first_index) + '.' + str(second_index) + '.0.' + structure_prefix + str(number_simulation) + '.' + str(first_index) + '.' + str(second_index) + '.0.config'
  N = sa.read_particle_number(name_config)

  # Read config
  names_config = []
  for j in range(N_samples):
    names_config.append(file_prefix + '.' + str(first_index) + '.' + str(second_index) + '.' + str(j) + '.' + structure_prefix + str(number_simulation) + '.' + str(first_index) + '.' + str(second_index) + '.' + str(j) + '.config')
  x = sa.read_config_list(names_config, print_name=True)
  num_frames = x.shape[0]
  print('x.shape    = ', x.shape)
  print('num_frames = ', num_frames)
  print('N          = ', N)
  print(' ')

