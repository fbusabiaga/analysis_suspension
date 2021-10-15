'''
Script to save xyz files.
'''
import numpy as np
import sys
import simulation_analysis as sa


if __name__ == '__main__':
  # Set parameters
  file_prefix = '/home/fbalboa/simulations/RigidMultiblobsWall/chiral/data/run700/run744/run744.1.0.0'
  structure = 'superellipsoid_run744.1.0.0'
  name_vertex = '/home/fbalboa/sfw/RigidMultiblobsWall/multi_bodies/examples/chiral/Structures/superellipsoid_Lg_1.368_r_3.9_N_26.vertex'
  num_frames = 10
  save_blobs = True
  save_dipole = False
  save_velocity = False

  # Read inputfile
  name_input = file_prefix + '.inputfile' 
  read = sa.read_input(name_input)

  # Get number of particles
  name_config = file_prefix + '.' + structure + '.config'
  N = sa.read_particle_number(name_config)

  # Set some parameters
  dt = float(read.get('dt')) 
  n_save = int(read.get('n_save'))
  dt_sample = dt * n_save
  eta = float(read.get('eta'))
  omega = float(read.get('omega'))
  print('file_prefix = ', file_prefix)
  print('file_config = ', name_config)
  print('dt          = ', dt)
  print('dt_sample   = ', dt_sample)
  print('n_save      = ', n_save)
  print('eta         = ', eta)
  print(' ')
  
  # Read config
  x = sa.read_config(name_config)
  t = np.arange(x.shape[0]) * dt
  print('total number of frames = ', x.shape[0])
  print('num_frames             = ', num_frames)

  # Read vertex
  r_vectors = sa.read_vertex(name_vertex)

  # Save xyz for blobs
  if save_blobs:
    name = file_prefix + '.' + structure + '.xyz'
    sa.save_xyz(x, r_vectors, name, num_frames=num_frames, letter=structure[0].upper())

  # Save velocity as xyz file
  if save_velocity:
    # Compute velocity
    velocity = sa.compute_velocities(x, dt=dt, frame_rate=frame_rate)
    print('velocity = ', velocity.shape)




  
