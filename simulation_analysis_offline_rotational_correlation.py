import numpy as np
import sys
import simulation_analysis as sa
import time


if __name__ == '__main__':
  # Set parameters
  file_prefix = '/home/fbalboa/simulations/RigidMultiblobsWall/rheology/data/run2000/run2161/run2161.3.0.0'
  files_method = 'sequence' # 'sequence'
  file_start = 0
  file_end = 50
  file_prefix_config = '/home/fbalboa/simulations/RigidMultiblobsWall/rheology/data/run2000/run2161/run2161.3.0.'
  file_suffix_config = '.star_run2161.3.0.' 
  # files_config = ['/home/fbalboa/simulations/RigidMultiblobsWall/rheology/data/run2000/run2165/run2165.0.0.0.star_run2165.0.0.0.config']
  suffix = '.star_run2161.*.dat'  
  simulation_number_start = 0
  simulation_number_end = 40
  num_frames = 10000
  num_frames_skip_fraction = 0

  # Start time counter
  time_start = time.time()
  
  # Read input file
  name_input = file_prefix + '.inputfile'
  print('name_input  = ', name_input)  
  read = sa.read_input(name_input)
  dt = float(read.get('dt')) 
  n_save = int(read.get('n_save'))
  dt_sample = dt * n_save

  # Get names config files
  if files_method == 'sequence':
    files_config = []
    for i in range(file_start, file_end + 1):
      name = file_prefix_config + str(i) + file_suffix_config + str(i) + '.config'
      files_config.append(name)
  
  # Get number of particles
  N = sa.read_particle_number(files_config[0])
  
  # Loop over config files
  x = []
  for j, name_config in enumerate(files_config):
    print('name_config = ', name_config)
    xj = sa.read_config(name_config)
    if j == 0 and xj.size > 0:
      x.append(xj)
    elif xj.size > 0:
      x.append(xj[1:])

  # Concatenate config files
  x = np.concatenate([xi for xi in x])
  num_frames = x.shape[0] if x.shape[0] < num_frames else num_frames
  if num_frames_skip_fraction > 0:
    num_frames_skip = num_frames // num_frames_skip_fraction
  else:
    num_frames_skip = 0
  print('total number of frames = ', x.shape[0])
  print('num_frames             = ', num_frames)
  print('num_frames_skip        = ', num_frames_skip)
  print('dt_sample              = ', dt_sample)
  print(' ')

  # Call msd
  name = file_prefix + '.rotational_correlation.dat'
  sa.rotational_correlation(x, dt_sample, np.array([1, 0, 0]), Corr_steps=num_frames, output_name=name)

  print('time = ', time.time() - time_start)
  
