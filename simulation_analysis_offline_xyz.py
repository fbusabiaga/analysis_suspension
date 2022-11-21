'''
Script to save xyz files.
'''
import numpy as np
import sys
import simulation_analysis as sa
from quaternion import Quaternion

if __name__ == '__main__':
  # Set parameters
  file_prefix = '/home/fbalboa/simulations/RigidMultiblobsWall/rheology/data/run2000/run2131/run2131.3.0.0'
  files_method = 'sequence' # 'sequence'
  file_start = 0
  file_end = 25
  file_prefix_config = '/home/fbalboa/simulations/RigidMultiblobsWall/rheology/data/run2000/run2131/run2131.3.0.'
  file_suffix_config = '.star_run2131.3.0.' 
  files_config = ['/home/fbalboa/simulations/RigidMultiblobsWall/rheology/data/run2000/run2131/run2131.3.0.0.star_run2131.3.0.0.config']
  structure = 'star'
  # name_vertex = '/home/fbalboa/simulations/RigidMultiblobsWall/rheology/data/superellipsoid_Lg_1.368_r_3.9_N_26.vertex'
  name_vertex = '/home/fbalboa/sfw/RigidMultiblobsWall/multi_bodies/examples/rheology/Structures/star_hook_N_25_a_0.05.vertex'
  num_frames = 200100
  n_save_xyz = 50
  save_blobs = True
  save_dipole = False
  save_velocity = False
  save_dat_index = []
  subset = 50

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
    blob_vector_constant = np.zeros((x.shape[1], r_vectors.shape[0]))
    for i in range(x.shape[1]):
      blob_vector_constant[i,:] = x[0,i,0]
    sa.save_xyz(x, r_vectors, name, num_frames=num_frames, letter=structure[0].upper(), blob_vector_constant=blob_vector_constant)
       
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
