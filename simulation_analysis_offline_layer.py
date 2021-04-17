import numpy as np
import sys
import simulation_analysis as sa


if __name__ == '__main__':
  # Set parameters
  file_prefix = '/home/fbalboa/simulations/RigidMultiblobsWall/rheology/data/run2000/run2001/run2001.1.0.0'
  name_input = '/home/fbalboa/simulations/RigidMultiblobsWall/rheology/data/run2000/run2001/run2001.1.0.0.inputfile'
  name_config = '/home/fbalboa/simulations/RigidMultiblobsWall/rheology/data/run2000/run2001/run2001.1.0.base.star.config'
  N_hist = 4
  
  # Read inputfile
  read = sa.read_input(name_input)

  # Set some parameters
  dt = float(read.get('dt')) 
  n_save = int(read.get('n_save'))
  dt_sample = dt * n_save
  eta = float(read.get('eta'))
  gamma_dot = float(read.get('gamma_dot'))
  L = np.fromstring(read.get('periodic_length'), sep=' ')
  wall_Lz = float(read.get('wall_Lz'))
  print('dt        = ', dt)
  print('dt_sample = ', dt_sample)
  print('n_save    = ', n_save)
  print('eta       = ', eta)
  print('gamme_dot = ', gamma_dot)
  print('L         = ', L)
  print('wall_Lz   = ', wall_Lz)
  print(' ')
  
  # Read config
  x = sa.read_config(name_config)
  num_frames = x.shape[0]
  num_frames_vel = num_frames - 1
  N = x.shape[1]
  N_avg = num_frames // N_hist
  print('num_frames = ', num_frames)
  print('N          = ', N)
  print('N_avg      = ', N_avg)
  print(' ')

  # Compute velocities
  v = sa.compute_velocities(x, dt=dt_sample)

  # Compute velocity histograms
  name = file_prefix + '.histogram_velocity'
  h = sa.compute_histogram_from_frames(x, v, column_sample=2, column_value=0, num_intervales=40, xmin=0, xmax=wall_Lz, N_avg=N_avg, file_prefix=name)

  # Compute viscosity from three last histograms
  eta_v1, _ = sa.compute_viscosity_from_profile(h[-1], gamma_dot=gamma_dot, eta_0=eta)
  eta_v2, _ = sa.compute_viscosity_from_profile(h[-2], gamma_dot=gamma_dot, eta_0=eta)
  eta_v3, _ = sa.compute_viscosity_from_profile(h[-3], gamma_dot=gamma_dot, eta_0=eta)

  # Compute average viscosity and error
  eta_mean = (eta_v1 + eta_v2 + eta_v3) / 3.0
  eta_std = np.sqrt(((eta_v1 - eta_mean)**2 + (eta_v2 - eta_mean)**2 + (eta_v3 - eta_mean)**2) / (3 * 2)) 
  print('eta_v1 = ', eta_v1)
  print('eta_v2 = ', eta_v2)
  print('eta_v3 = ', eta_v3)
  print('eta    = ', eta_mean, eta_std)

  
