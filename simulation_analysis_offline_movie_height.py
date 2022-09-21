'''
Compute height color maps. 
'''
import numpy as np
import sys
import simulation_analysis as sa
from quaternion import Quaternion

if __name__ == '__main__':
  # Set parameters
  file_prefix = '/home/fbalboa/simulations/RigidMultiblobsWall/chiral/data/run3000/run3007/run3007.5.7.0'
  files_config = ['/home/fbalboa/simulations/RigidMultiblobsWall/chiral/data/run3000/run3007/run3007.5.7.0.superellipsoid_run3007.5.7.0.config',
                  '/home/fbalboa/simulations/RigidMultiblobsWall/chiral/data/run3000/run3007/run3007.5.7.1.superellipsoid_run3007.5.7.1.config']
  structure = 'superellipsoid'
  num_frames = 10
  n_avg = 10
  nx = 25
  ny = 25
  x_ends = np.array([-30, 30])
  y_ends = np.array([-30, 30])  
  vmin=1
  vmax=2.5

  # Read inputfile
  name_input = file_prefix + '.inputfile' 
  read = sa.read_input(name_input)

  # Get number of particles
  N = sa.read_particle_number(files_config[0])

  # Set some parameters
  dt = float(read.get('dt')) 
  n_save = int(read.get('n_save'))
  dt_sample = dt * n_save
  eta = float(read.get('eta'))
  print('file_prefix = ', file_prefix)
  print('dt          = ', dt)
  print('dt_sample   = ', dt_sample)
  print('n_save      = ', n_save)
  print('eta         = ', eta)
  print(' ')

  # Loop over config files
  x = sa.read_config_list(files_config, print_name=True)
  num_frames = num_frames if num_frames < x.shape[0] else x.shape[0]

  # Get time
  print('total number of frames = ', x.shape[0])
  print('num_frames             = ', num_frames)
  print(' ')
  x = x[0:num_frames]
  t = np.arange(x.shape[0]) * dt_sample
  
  # Get color map height
  sa.map2d(x[:,:,0], x[:,:,1], x[:,:,2], centered=True, nx=nx, ny=ny, n_avg=n_avg, x_ends=x_ends, y_ends=y_ends, vmin=vmin, vmax=vmax, file_prefix=file_prefix + '.map_height')
  
  
