'''
Cluster analysis for the chiral simulations.
'''
import numpy as np
import sys
# import sklearn.cluster as skcl
import matplotlib.pyplot as plt
import matplotlib as mpl
import simulation_analysis as sa
from quaternion import Quaternion

if __name__ == '__main__':
  # Set parameters
  file_prefix = '/home/fbalboa/simulations/RigidMultiblobsWall/chiral/data/run3000/run3116/run3116.'
  # indices = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8], dtype=int)
  indices = np.array([8], dtype=int)
  # indices = np.array([7, 15, 25, 50, 100, 200, 400, 800, 1000], dtype=int)
  second_index = 0
  third_index = 0  
  N_samples = 1
  # N = np.array([7, 15, 25, 50, 100, 200, 400, 800, 1000], dtype=int)
  N = np.array([1000], dtype=int)
  file_prefix_config = '/home/fbalboa/simulations/RigidMultiblobsWall/chiral/data/run3000/run3116/run3116.'
  # file_suffix_config = ['.superellipsoid_N_7.config', '.superellipsoid_N_15.config', '.superellipsoid_N_25.config', '.superellipsoid_N_50.config', '.superellipsoid_N_100.config', '.superellipsoid_N_200.config', '.superellipsoid_N_400.config', '.superellipsoid_N_800.config', '.superellipsoid_N_1000.config']
  file_suffix_config = ['.superellipsoid_N_1000.config']
  files_config = None
  num_frames = 40000
  max_eps = 10
  n_save = 100
  axis_roll = 0
  axis_transverse = 1

  # Prepare array for data: index, tilt, v_roll, v_perp, v_z
  results = np.zeros((indices.size, 8))
  results[:,0] = indices
  results[:,1] = N

  for i, index in enumerate(indices):
    # Read inputfile 
    name_input = file_prefix + str(index) + '.' + str(second_index) + '.' + str(third_index) + '.inputfile'  
    read = sa.read_input(name_input) 
    
    # Read config
    names_config = []
    for j in range(0, N_samples):
      names_config.append(file_prefix + str(index) + '.' + str(second_index) + '.' + str(j) + file_suffix_config[i])
   
    # Get number of particles
    N = sa.read_particle_number(names_config[0])

    # Loop over config files
    x = sa.read_config_list(names_config, print_name=True)
    if num_frames < x.shape[0]:
      x = x[0:num_frames]
    skip_steps = x.shape[0] // 4
    
    # Get number of particles 
    N = sa.read_particle_number(names_config[0]) 
  
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
    print('max_eps     = ', max_eps)  
    print(' ')

    # Concatenate config files 
    t = np.arange(x.shape[0]) * dt_sample 
    print('total number of frames = ', x.shape[0])  
    print('num_frames             = ', num_frames)  
    print(' ')

    # Get labels of clusters
    name = file_prefix + str(index) + '.' + str(second_index) + '.' + str(third_index)
    labels = sa.cluster_detection_sklearn_optics(x, max_eps=max_eps, min_samples=5, n_save=100, file_prefix=name)

    # Compute center of mass of cluster 0
    x_cm = np.zeros((x.shape[0], 3))
    diameter = []
    for j, xj in enumerate(x):
      x_cm[j] = np.mean(xj[labels[j] == 0, 0:3], axis=0)

      # Compute diameter
      xj_cluster = xj[labels[j] == 0, 0:3]
      rx = xj_cluster[:,0]
      ry = xj_cluster[:,1]
      rz = xj_cluster[:,2]  
      dx = rx - rx[:,None]
      dy = ry - ry[:,None]
      dz = rz - rz[:,None]
      d = np.sqrt(dx*dx + dy*dy + dz*dz)
      diameter.append(np.max(d))

    # Save radius
    radius = np.mean(diameter) / 2
    radius_std = np.std(diameter) / 2
    results[i,5] = radius
    results[i,6] = radius_std
      
    # Save center of mass
    name = file_prefix + '.r_cm.dat'
    tmp = np.zeros((x_cm.shape[0], 4))
    tmp[:,0] = t
    tmp[:,1:] = x_cm
    np.savetxt(name, tmp)

    def func(t, x0, slope):
      return x0 + slope * t

    # Plot velocity roll
    name = file_prefix + str(index) + '.' + str(second_index) + '.0.fit_velocity_roll.png'    
    p0 = np.array([0, (x_cm[-1, axis_roll] - x_cm[0, axis_roll]) / t[-1]])
    popt_roll, pcov_roll, R2_roll = sa.nonlinear_fit(t[skip_steps:], x_cm[skip_steps:,axis_roll], func, sigma=None, p0=p0, save_plot_name=name)

    # Plot velocity transverse
    name = file_prefix + str(index) + '.' + str(second_index) + '.0.fit_velocity_transverse.png'
    p0 = np.array([0, (x_cm[-1, axis_transverse] - x_cm[0, axis_transverse]) / t[-1]])
    popt_transverse, pcov_transverse, R2_transverse = sa.nonlinear_fit(t[skip_steps:], x_cm[skip_steps:,axis_transverse], func, sigma=None, p0=p0, save_plot_name=name)
    
    # Plot velocity z
    name = file_prefix + str(index) + '.' + str(second_index) + '.0.fit_velocity_z.png'
    p0 = np.array([0, (x_cm[-1, 2] - x_cm[0, 2]) / t[-1]])
    popt_z, pcov_z, R2_z = sa.nonlinear_fit(t[skip_steps:], x_cm[skip_steps:,2], func, sigma=None, p0=p0, save_plot_name=name)
    print('popt_z = ', popt_z)
    
    # Print velocity
    print('velocity center of mass = ', popt_roll[1], popt_transverse[1], popt_z[1])






    # Save data
    results[i,2] = popt_roll[1]
    results[i,3] = popt_transverse[1]
    results[i,4] = popt_z[1]
    results[i,7] = popt_z[0]


  # Save to file
  name = file_prefix + str(indices[0]) + '-' + str(indices[-1]) + '.' + str(second_index) + '.' + str(0) + '.velocities_cm.dat'  
  header = 'Columns: indices, N, v_parallel, v_perp, v_z, radius, radius_std, height'
  np.savetxt(name, results, header=header)
  
