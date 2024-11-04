'''
Set files to restart a simulation.
'''
import numpy as np
import sys
import subprocess
import simulation_analysis as sa


if __name__ == '__main__':
  # Set parameters
  run_folder = '/home/fbalboa/kk/'
  file_prefix = '/home/fbalboa/simulations/RigidMultiblobsWall/chiral/data/run3000/run3112/run3112.'
  simulation_index = 3112
  first_index = 1000
  second_index = 10
  third_index_last = 2
  config_suffix = '.superellipsoid_N_1000.config'
  clones_sufix = '.superellipsoid_N_1000.'
  N_samples = 5

  # Read inputfile
  name_input = file_prefix + str(first_index) + '.' + str(second_index) + '.' + str(0) + '.inputfile'  
  read = sa.read_input(name_input)
  n_save = int(read.get('n_save')) 

  # Copy input file
  data_file = run_folder + 'data.main.' + str(3112) + '.' + str(first_index) + '.' + str(second_index) + '.' + str(third_index_last + 1)
  subprocess.run(['cp', name_input, data_file])

  # Get number of particles
  file_config = file_prefix + str(first_index) + '.' + str(second_index) + '.' + str(third_index_last) + config_suffix
  N = sa.read_particle_number(file_config)
  print('N = ', N)

  # Read config file
  x = sa.read_config(file_config)
  print('x.shape[0] = ', x.shape[0])
  
  # Crop config file
  num_lines = x.shape[0] * (N+1)
  print('num_lines         = ', num_lines)
  with open('tmp.dat', 'w') as f_handle:
    subprocess.call(['head', '-' + str(num_lines), file_config], stdout=f_handle)
  subprocess.run(['mv', 'tmp.dat', file_config])
  
  # Read all config files
  names_config = []
  for j in range(0, N_samples):
    names_config.append(file_prefix + str(first_index) + '.' + str(second_index) + '.' + str(j) + config_suffix)
  x = sa.read_config_list(names_config, print_name=True)
  
  # Save additional clones file
  clones_name = file_prefix + str(first_index) + '.' + str(second_index) + '.' + str(third_index_last + 1) + clones_sufix + str((x.shape[0] - 1) * n_save).zfill(8) + '.clones'
  with open('tmp.dat', 'w') as f_handle:
    subprocess.call(['tail', '-' + str(N + 1), file_config], stdout=f_handle)
  subprocess.run(['mv', 'tmp.dat', clones_name])
  print('clones_name = ', clones_name)





