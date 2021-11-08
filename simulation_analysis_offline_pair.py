'''
Script to save escape times for a two particle cluster.
'''
import numpy as np
import sys
import simulation_analysis as sa


if __name__ == '__main__':
  # Set parameters
  file_prefix = '/home/fbalboa/simulations/RigidMultiblobsWall/rheology/data/run2000/run2020/run2020.0.0'
  structure_prefix = 'star_run2020.0.0'
  num_frames = 50001
  N = 100

  # Prepare output, columns: number of frames
  escape_times = np.zeros(N)

  # Loop over simulations
  for i in range(N):    
    # Read inputfile
    name_input = file_prefix + '.' + str(i) + '.inputfile' 
    read = sa.read_input(name_input)
    dt = float(read.get('dt')) 
    n_save = int(read.get('n_save'))
    dt_sample = dt * n_save

    # Set some parameters
    dt = float(read.get('dt')) 
    n_save = int(read.get('n_save'))
    dt_sample = dt * n_save
    eta = float(read.get('eta'))
    gamma_dot = float(read.get('gamma_dot'))

  
    # Read config
    name_config = file_prefix + '.' + str(i) + '.' + structure_prefix + '.' + str(i) + '.config'
    x = sa.read_config(name_config)
    print('x = ', x.shape)
    escape_times[i] = x.shape[0] * dt_sample
    
  # Save result 
  name = file_prefix + '.0-' + str(N-1) + '.escape_times.dat'
  np.savetxt(name, escape_times)

  # Compute histogram
  name = file_prefix + '.0-' + str(N-1) + '.histogram.escape_times.dat'
  xmax = np.sort(escape_times)[-1] * 1.25
  print('xmax = ', xmax)
  sa.compute_histogram(escape_times.reshape((N, 1)), num_intervales=25, xmin=0, xmax=xmax, name=name)

  
