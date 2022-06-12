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

try:
  sys.path.append('../RigidMultiblobsWall/')
  from visit import visit_writer as visit_writer
except ImportError as e:
  print(e)
  pass


if __name__ == '__main__':
  # Set parameters
  file_prefix = '/home/fbalboa/simulations/RigidMultiblobsWall/chiral/data/run3000/run3007/run3007.5.3.0'
  files_config = ['/home/fbalboa/simulations/RigidMultiblobsWall/chiral/data/run3000/run3007/run3007.5.3.0.superellipsoid_run3007.5.3.0.config',
                  '/home/fbalboa/simulations/RigidMultiblobsWall/chiral/data/run3000/run3007/run3007.5.3.1.superellipsoid_run3007.5.3.1.config']
  structure = 'superellipsoid'
  num_frames = 10
  tilt_vtk = True
  grid = np.array([-1000, 1000, 5, -1000, 1000, 5, -1000, 1000, 5])

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
      grid_omega_z = np.zeros(grid_coor.shape[0])
      grid_omega_B = np.zeros_like(grid_omega_z)
      grid_omega_perp = np.zeros_like(grid_omega_z)
      grid_omega_norm = np.zeros_like(grid_omega_z)
      grid_omega_count = np.zeros_like(grid_omega_z)
      B0_hat = np.array([np.cos(omega * t[i]), np.sin(omega * t[i]), 0])
      B = np.dot(R_B, B0_hat)

      # Compute center of mass
      xi = x[i]
      r_cm = np.mean(xi[:,0:3], axis=0)
      
      for j, vij in enumerate(vi):
        # Get angular velocity direction
        omega_norm = np.linalg.norm(vij[3:])
        omega_vec_hat = vij[3:] / omega_norm
        
        # Save components of angular velocity in rotating frame
        omega_z = np.dot(omega_vec_hat, z)
        omega_B = np.dot(omega_vec_hat, B)
        omega_perp = np.dot(omega_vec_hat, np.cross(z, B))

        # Get cell index
        ix = int((x[i,j,0] - grid[0,0] - r_cm[0]) / dx_grid[0])
        iy = int((x[i,j,1] - grid[0,1] - r_cm[1]) / dx_grid[1])
        iz = int((x[i,j,2] - grid[0,2] - r_cm[2]) / dx_grid[2])
        index = iz * grid_points[0] * grid_points[1] + iy * grid_points[0] + ix

        # Save info
        if (ix >= 0 and ix < grid_length[0] and
            iy >= 0 and iy < grid_length[0] and
            iz >= 0 and iz < grid_length[2]):
          grid_omega_count[index] += 1
          grid_omega_z[index] += omega_z
          grid_omega_B[index] += omega_B
          grid_omega_perp[index] += omega_perp
          grid_omega_norm[index] += omega_norm
          
      # Normalize saved variables
      sel = grid_omega_count > 0
      grid_omega_z[sel] /= grid_omega_count[sel]
      grid_omega_B[sel] /= grid_omega_count[sel]
      grid_omega_perp[sel] /= grid_omega_count[sel]
      grid_omega_norm[sel] /= grid_omega_count[sel]

    # Prepara data for VTK writer 
    variables = [grid_omega_count]
    dims = np.array([grid_points[0]+1, grid_points[1]+1, grid_points[2]+1], dtype=np.int32) 
    nvars = 1
    vardims = np.array([1])
    centering = np.array([0])
    varnames = ['omega_count\0']
    name = file_prefix + '.omega_field.vtk'
    grid_x = grid_x - dx_grid[0] * 0.5
    grid_y = grid_y - dx_grid[1] * 0.5
    grid_z = grid_z - dx_grid[2] * 0.5
    grid_x = np.concatenate([grid_x, [grid[1,0]]])
    grid_y = np.concatenate([grid_y, [grid[1,1]]])
    grid_z = np.concatenate([grid_z, [grid[1,2]]])
        
    # Write velocity field
    visit_writer.boost_write_rectilinear_mesh(name,      # File's name
                                              0,         # 0=ASCII,  1=Binary
                                              dims,      # {mx, my, mz}
                                              grid_x,     # xmesh
                                              grid_y,     # ymesh
                                              grid_z,     # zmesh
                                              nvars,     # Number of variables
                                              vardims,   # Size of each variable, 1=scalar, velocity=3*scalars
                                              centering, # Write to cell centers of corners
                                              varnames,  # Variables' names
                                              variables) # Variables
