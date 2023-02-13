import numpy as np
import sys
import subprocess
import simulation_analysis as sa


if __name__ == '__main__':
  # Set parameters
  file_prefix = '/home/fbalboa/simulations/RigidMultiblobsWall/rheology/data/run2000/run2130/run2130'
  inputfile_prefix = '/workspace/scratch/users/fbalboa/sfw/RigidMultiblobsWall/multi_bodies/examples/rheology/data.main.2130' 
  structure_prefix = 'star_run'
  first_index = 1
  second_index = 0
  rerun_n = '.rerun_0.'
  number_simulation = 2130
  N_samples = 36
  
  # Get number of particles
  name_config = file_prefix + '.' + str(first_index) + '.' + str(second_index) + '.' + str(N_samples - 1) + '.' + structure_prefix + str(number_simulation) + '.' + str(first_index) + '.' + str(second_index) + '.' + str(N_samples - 1) + '.config'
  N = sa.read_particle_number(name_config)

  # Copy input file
  name_input = file_prefix + '.' + str(first_index) + rerun_n + str(N_samples - 2) + '.inputfile'
  inputfile_name = inputfile_prefix + '.' + str(first_index) + rerun_n + str(N_samples - 1) 
  subprocess.run(['cp', name_input, inputfile_name])

  if True:
    # First read 
    name_input = file_prefix + '.' + str(first_index) + rerun_n + str(N_samples - 2) + '.inputfile'
    read = sa.read_input(name_input)
    offset = int(read.get('rerun_offset')) 

    # Read config
    for j in range(N_samples - 1, N_samples):
      names_config = [file_prefix + '.' + str(first_index) + '.' + str(second_index) + '.' + str(j) + '.' + structure_prefix + str(number_simulation) + '.' + str(first_index) + '.' + str(second_index) + '.' + str(j) + '.config']
      x = sa.read_config_list(names_config, print_name=True)
      num_frames = x.shape[0]
      offset += num_frames 
      print('offset    = ', offset)
      print(' ')
