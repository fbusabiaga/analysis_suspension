'''
Compute tilt angle from angular velocity.

Save angular vector components in the rotating frame given by the vectors
z = (0,0,1)
B = magnetic field direction
z \times B = perpendicular directions.
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
  num_frames = 10000
  tilt_vtk = False
  grid = np.array([-100, 100, 10, -100, 100, 10, 0, 0, 1])

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
  omega = float(read.get('omega'))
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
  if num_frames < x.shape[0]:
    x = x[0:num_frames]

  # Concatenate config files
  t = np.arange(x.shape[0]) * dt_sample
  print('total number of frames = ', x.shape[0])
  print('num_frames             = ', num_frames)
  print(' ')
    
  # Compute velocity
  velocity = sa.compute_velocities(x, dt=dt_sample)

  # Open output file
  outputname = file_prefix + '.tilt_omega.dat' 
  f_handle = open(outputname, 'w')
  f_handle.write('# Angular velocity direction in rotating frame vector given by the vectors z, B and z \times B. \n')
  f_handle.write('# Columns: time, omega_z, omega_B, omega_{z \times B}, omega_magnitude. \n')

  # Compute tilt angle versus time
  z = np.array([0, 0, 1.0])
  for i, vi in enumerate(velocity):
    B0_hat = np.array([np.cos(omega * t[i]), np.sin(omega * t[i]), 0])
    B = np.dot(R_B, B0_hat)

    # Get angular velocity direction and magnitude
    omega_norm = np.linalg.norm(vi[:,3:], axis=1)
    omega_vec_hat = vi[:,3:] / omega_norm[:,None]      
    omega_z = np.einsum('ij,j->', omega_vec_hat, z) / vi.shape[0]
    omega_B = np.einsum('ij,j->', omega_vec_hat, B) / vi.shape[0]
    omega_perp = np.einsum('ij,j->', omega_vec_hat, np.cross(z,B)) / vi.shape[0]
    omega_norm_avg = np.einsum('i->', omega_norm) / vi.shape[0]

    # Save time value
    f_handle.write(str(i * dt_sample) + ' ' +  str(omega_z) + ' ' +  str(omega_B) + ' ' +  str(omega_perp) + ' ' +  str(omega_norm_avg) + '\n')

  # Save tilt to vtk file
  if tilt_vtk:
    grid = np.reshape(grid, (3,3)).T
    grid_length = grid[1] - grid[0]
    grid_points = np.array(grid[2], dtype=np.int32)
    num_points = grid_points[0] * grid_points[1] * grid_points[2]

    # Set grid coordinates
    dx_grid = grid_length / grid_points
    grid_x = np.array([grid[0,0] + dx_grid[0] * (x+0.5) for x in range(grid_points[0])])
    grid_y = np.array([grid[0,1] + dx_grid[1] * (x+0.5) for x in range(grid_points[1])])
    grid_z = np.array([grid[0,2] + dx_grid[2] * (x+0.5) for x in range(grid_points[2])])
    # Be aware, x is the fast axis.
    zz, yy, xx = np.meshgrid(grid_z, grid_y, grid_x, indexing = 'ij')
    grid_coor = np.zeros((num_points, 3))
    grid_coor[:,0] = np.reshape(xx, xx.size)
    grid_coor[:,1] = np.reshape(yy, yy.size)
    grid_coor[:,2] = np.reshape(zz, zz.size)
 
    for i, vi in enumerate(velocity):
      # Create vector for omega orientation
      grid_omega_z = np.zeros_like(grid_coor)
      grid_omega_B = np.zeros_like(grid_coor)
      grid_omega_perp = np.zeros_like(grid_coor)
      grid_omega_norm = np.zeros_like(grid_coor) 
      B0_hat = np.array([np.cos(omega * t[i]), np.sin(omega * t[i]), 0])
      B = np.dot(R_B, B0_hat)

      # Compute center of mass
      xi = x[i]
      r_cm = np.mean(xi[:,0:3], axis=0)
      print('r_cm = ', r_cm)
      
      for j, vij in enumerate(vi):
        # Get angular velocity direction
        omega_norm = np.linalg.norm(vij[3:])
        omega_vec_hat = vij[3:] / omega_norm
        
        # Save components of angular velocity in rotating frame
        omega_z = np.dot(omega_vec_hat, z)
        omega_B = np.dot(omega_vec_hat, B)
        omega_perp = np.dot(omega_vec_hat, np.cross(z, B))
        omega_norm_avg = omega_norm

        # # Get cell index
        # xij = x[i,j]
        # ix = xij[0] / 

      # Save time value
      omega_z /= vi.shape[0]
      omega_B /= vi.shape[0]
      omega_perp /= vi.shape[0]
      omega_norm_avg /= vi.shape[0]
