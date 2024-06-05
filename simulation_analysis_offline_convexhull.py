'''
This code compute the volume of a suspension in box using the convex hull.

If use_auxiliar_points == True the code add six auxiliar points to compute the convex hull.
'''

import numpy as np
import scipy.spatial as scsp
import sys
import simulation_analysis as sa
from quaternion import Quaternion

if __name__ == '__main__':
  # Set parameters
  file_prefix = '/home/fbalboa/simulations/RigidMultiblobsWall/sedimentation/data/dipankar/run4/run_sedimentation225'
  files_method = 'File' # 'sequence'
  file_start = 0
  file_end = 0
  files_config = ['/home/fbalboa/simulations/RigidMultiblobsWall/sedimentation/data/dipankar/run4/run_sedimentation225.colloids225.config']
  tilt = np.pi * 0.25
  use_auxiliar_points = True
  Lx_min = 0
  Lx_max = 7
  Ly_min = 0
  Ly_max = 7
  Lz_min = -15
  Lz_max = 7  
  num_frames = 10000
  
  # Get names config files
  if files_method == 'sequence':
    files_config = []
    for i in range(file_start, file_end + 1):
      name = file_prefix_config + str(i) + file_suffix_config + str(i) + '.config'
      files_config.append(name)
  
  # Read inputfile
  name_input = file_prefix + '.inputfile' 
  read = sa.read_input(name_input)

  # Get number of particles
  N = sa.read_particle_number(files_config[0])

  # Set some parameters
  dt = float(read.get('dt')) 
  n_save = int(read.get('n_save'))
  eta = float(read.get('eta'))
  print('file_prefix = ', file_prefix)
  print('dt          = ', dt)
  print('n_save      = ', n_save)
  print('eta         = ', eta)
  print(' ')

  # Loop over config files
  x = sa.read_config_list(files_config, print_name=True)

  # Concatenate config files
  print('total number of frames = ', x.shape[0])
  print('num_frames             = ', num_frames)
  print(' ')

  # Open convex hull file
  name = file_prefix + '.convex_hull.xyz'
  f_convexHull = open(name, 'w')

  # Loop over time steps
  xi_extended = np.zeros((x.shape[1] + 6, 3)) if use_auxiliar_points else np.zeros((x.shape[1] + 0, 3))
  result = np.zeros((x.shape[0], 2))
  result[:,0] = np.arange(x.shape[0]) * dt * n_save
  for ti, xi in enumerate(x):
    xi_extended[0:N] = xi[:,0:3]

    # Get vertical coordinates
    height = xi[:,2] * np.cos(tilt) - xi[:,0] * np.sin(tilt)
    perpendicular = xi[:,0] * np.cos(tilt) + xi[:,2] * np.sin(tilt)

    # Get highest particle and add fake particles at the same height 
    index = np.argmax(height)
    height_box = height[index] * np.cos(tilt) + Lx_max * np.sin(tilt)
    if use_auxiliar_points:
      xi_extended[N+0] = np.array([Lx_max, Ly_min, height_box])
      xi_extended[N+1] = np.array([Lx_max, Ly_max, height_box])

    # Get perpendicular most particle and add fake particles at the same perpendicular distance
    index = np.argmin(xi[:,0])
    if use_auxiliar_points:
      xi_extended[N+2] = np.array([xi[index,0], Ly_min, Lz_min])
      xi_extended[N+3] = np.array([xi[index,0], Ly_max, Lz_min])

    # Add bottom particles
    if use_auxiliar_points:
      xi_extended[N+4] = np.array([Lx_max, Ly_min, Lz_min])
      xi_extended[N+5] = np.array([Lx_max, Ly_max, Lz_min])

    # Create convex hull and save volume
    qH = scsp.ConvexHull(xi_extended)
    volume = qH.volume
    result[ti, 1] = volume
    print('ti, volume = ', ti, volume)

    # Save convex hull
    vertices = np.zeros((len(qH.vertices), 4))
    vertices[:,1:4] = xi_extended[qH.vertices]
    f_convexHull.write(str(vertices.shape[0]) + '\n#\n')
    np.savetxt(f_convexHull, vertices)
    
  # At the end you have all the dirty volume
  name = file_prefix + '.dirty_volume.dat'
  header = 'Columns: time, dirty volume'
  np.savetxt(name, result, header=header)
  
