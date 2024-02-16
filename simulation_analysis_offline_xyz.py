'''
Script to save xyz files.
'''
import numpy as np
import sys
import simulation_analysis as sa
from quaternion import Quaternion

if __name__ == '__main__':
  # Set parameters
  file_prefix = '/home/fbalboa/simulations/RigidMultiblobsWall/FreeSlip/data/run0/run6/run6.5.0.0'
  files_method = 'File' # 'sequence'
  file_start = 0
  file_end = 0
  file_prefix_config = None
  file_suffix_config = None
  files_config = ['/home/fbalboa/simulations/RigidMultiblobsWall/FreeSlip/data/run0/run6/run6.5.0.0.colloids.config']
  structure = 'colloids'
  name_vertex = '/home/fbalboa/sfw/RigidMultiblobsWall/multi_bodies/Structures/shell_N_42_Rg_0_8913_Rh_1.vertex'
  num_frames = 10000
  n_save_xyz = 1
  save_blobs = True
  save_dipole = True
  save_velocity = False
  save_dat_index = []

  # Get names config files
  if files_method == 'sequence':
    files_config = []
    for i in range(file_start, file_end + 1):
      name = file_prefix_config + str(i) + file_suffix_config + str(i) + '.config'
      files_config.append(name)
  
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
  dt_sample = dt * n_save * n_save_xyz
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
  x = x[0::n_save_xyz]

  # Concatenate config files
  t = np.arange(x.shape[0]) * dt_sample
  print('total number of frames = ', x.shape[0])
  print('num_frames             = ', num_frames)
  print(' ')

  # Save xyz for blobs
  if save_blobs:
    name = file_prefix + '.' + structure + '.xyz'
    num_blobs = r_vectors.shape[0] * x.shape[1]
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
    dipole = np.zeros((x.shape[1], 3))
    dipole[:] = dipole_0
    B1 = np.zeros((x.shape[0], 3))
    if False:
      B0 = float(read.get('B0'))
      omega = float(read.get('omega'))
      B1 = np.zeros((x.shape[0], 3))
      B1[:,0] = B0 * np.cos(omega * t)
      B1[:,1] = B0 * np.sin(omega * t)
    else:
      B0 = np.fromstring(read.get('B0') or '0 0 0', sep=' ')
      omega = np.fromstring(read.get('omega') or '0 0 0', sep=' ')
      phi = np.fromstring(read.get('phi') or '0 0 0', sep=' ')
      B1[:] = B0[None,:] * np.cos(omega[None,:] * t[:,None] + phi[None,:])   
    B = np.einsum('ij,kj->ki', R_B, B1)
    name = file_prefix + '.' + structure + '.dipole.xyz'    
    print('B0    = ', B0)
    print('omega = ', omega)
    print('name  = ', name)
    sa.save_xyz(x, np.zeros(3), name, num_frames=num_frames, letter=structure[0].upper(), body_frame_vector=dipole, global_vector=B, header='Columns: r, dipole, magnetic field')
