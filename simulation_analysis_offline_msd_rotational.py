import numpy as np
import sys
import simulation_analysis as sa


if __name__ == '__main__':
  # Set parameters
  file_prefix = '/workspace/scratch/users/fbalboa/simulations/RigidMultiblobsWall/rheology/data/run2000/run2120/run2120.0.0.0'
  files_config = ['/workspace/scratch/users/fbalboa/simulations/RigidMultiblobsWall/rheology/data/run2000/run2120/run2120.0.0.0.star_run2120.0.0.0.config'] 
  num_frames = 200
  num_frames_skip_fraction = 0
  
  # Read input file
  name_input = file_prefix + '.inputfile'
  print('name_input  = ', name_input)  
  read = sa.read_input(name_input)
  dt = float(read.get('dt')) 
  n_save = int(read.get('n_save'))
  dt_sample = dt * n_save
  
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
  name = file_prefix + '.msd_rotational.dat'
  sa.msd_rotational(x, dt_sample, MSD_steps=num_frames, output_name=name)


  
