'''
Script to save xyz files.
'''
import numpy as np
import sys
import simulation_analysis as sa
from quaternion import Quaternion

if __name__ == '__main__':
  # Set parameters
  file_prefix = '/home/fbalboa/simulations/RigidMultiblobsWall/chiral/data/run3000/run3007/run3007.5.3.0'
  files_config = ['/home/fbalboa/simulations/RigidMultiblobsWall/chiral/data/run3000/run3007/run3007.5.3.0.superellipsoid_run3007.5.3.0.config',
                  '/home/fbalboa/simulations/RigidMultiblobsWall/chiral/data/run3000/run3007/run3007.5.3.1.superellipsoid_run3007.5.3.1.config']
  structure = 'superellipsoid'
  name_vertex = '/home/fbalboa/sfw/RigidMultiblobsWall/multi_bodies/examples/chiral/Structures/superellipsoid_Lg_1.368_r_3.9_N_26.vertex'
  num_frames = 1000
  save_blobs = True
  save_dipole = True
  save_velocity = False
  save_dat_index = []

  # Read inputfile
  name_input = file_prefix + '.inputfile' 
  read = sa.read_input(name_input)

  # Read vertex
  r_vectors = sa.read_vertex(name_vertex)

  # Get number of particles
  N = sa.read_particle_number(files_config[0])

  # Set some parameters
  dt = float(read.get('dt')) 
  n_save = int(read.get('n_save'))
  dt_sample = dt * n_save
  eta = float(read.get('eta'))
  if 'quaternion_B' in read:
    print('quaternion_B = ', read.get('quaternion_B'))
    quaternion_B = Quaternion(np.fromstring(read.get('quaternion_B'), sep=' ') / np.linalg.norm(np.fromstring(read.get('quaternion_B'), sep=' ')))
    R_B = quaternion_B.rotation_matrix()
  print('file_prefix = ', file_prefix)
  print('dt          = ', dt)
  print('dt_sample   = ', dt_sample)
  print('n_save      = ', n_save)
  print('eta         = ', eta)
  print(' ')

  # Loop over config files
  x = sa.read_config_list(files_config, print_name=True)

  # Concatenate config files
  t = np.arange(x.shape[0]) * dt_sample
  print('total number of frames = ', x.shape[0])
  print('num_frames             = ', num_frames)
  print(' ')

  # Save xyz for blobs
  if save_blobs:
    name = file_prefix + '.' + structure + '.xyz'
    sa.save_xyz(x, r_vectors, name, num_frames=num_frames, letter=structure[0].upper())
       
  # Save velocity as xyz file
  if save_velocity:
    # Compute velocity
    name = file_prefix + '.' + structure + '.velocity.xyz'    
    velocity = sa.compute_velocities(x, dt=dt_sample)
    sa.save_xyz(x, np.zeros(3), name, num_frames=num_frames, letter=structure[0].upper(), body_vector=velocity, header='Columns: r, velocity')

  # Save dat file
  if save_dat_index is not None:
    for i in save_dat_index:
      name = file_prefix + '.' + structure + '.' + str(i) + '.dat'
      sa.save_dat(x, t, i, name)
      
  # Save dipole as xyz file
  if save_dipole:
    dipole_0 = np.fromstring(read.get('mu'), sep=' ')
    B0 = float(read.get('B0'))
    omega = float(read.get('omega'))
    dipole = np.zeros((x.shape[1], 3))
    dipole[:] = dipole_0
    B_xy = np.zeros((x.shape[0], 3))
    B_xy[:,0] = B0 * np.cos(omega * t)
    B_xy[:,1] = B0 * np.sin(omega * t)
    B = np.einsum('ij,kj->ki', R_B, B_xy)  
    name = file_prefix + '.' + structure + '.dipole.xyz'    
    print('B0    = ', B0)
    print('omega = ', omega)
    print('name  = ', name)
    sa.save_xyz(x, np.zeros(3), name, num_frames=num_frames, letter=structure[0].upper(), body_frame_vector=dipole, global_vector=B, header='Columns: r, dipole, magnetic field')
