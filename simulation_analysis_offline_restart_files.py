'''
Set files to restart a simulation.
'''
import numpy as np
import sys
import subprocess
import simulation_analysis as sa


if __name__ == '__main__':
  # Set parameters
  index = '6.0.8'
  index_next = '6.0.9'
  file_prefix = '/workspace/scratch/users/fbalboa/simulations/RigidMultiblobsWall/rheology/data/run2000/run2121/run2121.' + index
  file_config = '/workspace/scratch/users/fbalboa/simulations/RigidMultiblobsWall/rheology/data/run2000/run2121/run2121.' + index + '.star_run2121.' + index + '.config'
  inputfile_name = '/workspace/scratch/users/fbalboa/sfw/RigidMultiblobsWall/multi_bodies/examples/rheology/data.main.2121.' + index_next
  clones_name = '/workspace/scratch/users/fbalboa/sfw/RigidMultiblobsWall/multi_bodies/examples/rheology/Structures/star_run2121.' + index_next + '.clones'

  # Read inputfile
  name_input = file_prefix + '.inputfile' 
  read = sa.read_input(name_input)

  # Copy input file
  subprocess.run(['cp', name_input, inputfile_name])

  # Get number of particles
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

  # Set init config
  with open('tmp.dat', 'w') as f_handle:
    subprocess.call(['tail', '-' + str(N + 1), file_config], stdout=f_handle)
  subprocess.run(['mv', 'tmp.dat', clones_name])
  



