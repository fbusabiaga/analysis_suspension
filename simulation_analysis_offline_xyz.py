'''
Script to save xyz files.
'''
import numpy as np
import sys
import simulation_analysis as sa


if __name__ == '__main__':
  # Set parameters
  file_prefix = '/home/fbalboa/simulations/RigidMultiblobsWall/chiral/data/run2000/run2103/run2103.1.0.0'
  structure = 'shell'
  name_vertex = '/home/fbalboa/sfw/RigidMultiblobsWall/multi_bodies/Structures/blob.vertex'
  num_frames = 2500
  save_blobs = False
  save_dipole = True
  save_velocity = True

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
  t = np.arange(x.shape[0]) * dt_sample
  print('total number of frames = ', x.shape[0])
  print('num_frames             = ', num_frames)

  # Read vertex
  r_vectors = sa.read_vertex(name_vertex)

  # Save xyz for blobs
  if save_blobs:
    name = file_prefix + '.' + structure + '.xyz'
    sa.save_xyz(x, r_vectors, name, num_frames=num_frames, letter=structure[0].upper())

  # Save dipole as xyz file
  if save_dipole:
    dipole_0 = np.fromstring(read.get('mu'), sep=' ')
    B0 = float(read.get('B0'))
    omega = float(read.get('omega'))
    dipole = np.zeros((x.shape[1], 3))
    dipole[:] = dipole_0
    B = np.zeros((x.shape[0], 3))
    B[:,0] = B0 * np.cos(omega * t)
    B[:,1] = B0 * np.sin(omega * t)
    name = file_prefix + '.' + structure + '.dipole.xyz'    
    sa.save_xyz(x, np.zeros(3), name, num_frames=num_frames, letter=structure[0].upper(), body_frame_vector=dipole, global_vector=B, header='Columns: r, dipole, magnetic field')
    
  # Save velocity as xyz file
  if save_velocity:
    # Compute velocity
    name = file_prefix + '.' + structure + '.velocity.xyz'    
    velocity = sa.compute_velocities(x, dt=dt_sample)
    sa.save_xyz(x, np.zeros(3), name, num_frames=num_frames, letter=structure[0].upper(), body_vector=velocity, header='Columns: r, velocity')




  
