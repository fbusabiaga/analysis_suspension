import numpy as np
import sys
import simulation_analysis as sa


if __name__ == '__main__':
  # Set parameters
  file_prefix = '/mnt/home/fbalboausabiaga/symlinks/ceph/sfw/RigidMultiblobsWall/rheology/data/run2000/run2102/run2102'
  second_index = 0
  indices = np.arange(1, 7, dtype=int)
  N_hist = 4
  number_simulation = 2102
  N_samples = 2
  print('indices = ', indices)

  # Prepare viscosity file
  eta_files = np.zeros((len(indices), 10))
  eta_files[:,0] = number_simulation
  
  # Loop over files
  for k, i in enumerate(indices):  
    name_input = file_prefix + '.' + str(i) + '.' + str(second_index) + '.0.inputfile'
    print('name_input  = ', name_input)        

    # Read inputfile
    read = sa.read_input(name_input)

    # Get number of particles
    name_config = file_prefix + '.' + str(i) + '.' + str(second_index) + '.0.star_run' + str(number_simulation) + '.' + str(i) + '.' + str(second_index) + '.0.config'
    N = sa.read_particle_number(name_config)

    # Set some parameters
    dt = float(read.get('dt')) 
    n_save = int(read.get('n_save'))
    dt_sample = dt * n_save
    eta = float(read.get('eta'))
    gamma_dot = float(read.get('gamma_dot'))
    L = np.fromstring(read.get('periodic_length'), sep=' ')
    wall_Lz = float(read.get('wall_Lz'))
    volume = L[0] * L[1] * wall_Lz
    number_density = N / volume
    print('dt        = ', dt)
    print('dt_sample = ', dt_sample)
    print('n_save    = ', n_save)
    print('eta       = ', eta)
    print('gamme_dot = ', gamma_dot)
    print('L         = ', L)
    print('wall_Lz   = ', wall_Lz)
    print(' ')
  
    # Read config
    x = []
    for j in range(N_samples):
      name_config = file_prefix + '.' + str(i) + '.' + str(second_index) + '.' + str(j) + '.star_run' + str(number_simulation) + '.' + str(i) + '.' + str(second_index) + '.' + str(j) + '.config'
      print('name_config = ', name_config)

      xj = sa.read_config(name_config)
      if j == 0 and xj.size > 0:
        x.append(xj)
      elif xj.size > 0:
        x.append(xj[1:])
      
    x = np.concatenate([xi for xi in x])
    num_frames = x.shape[0]
    num_frames_vel = num_frames - 1
    N_avg = (num_frames-1) // N_hist
    print('x.shape    = ', x.shape)
    print('num_frames = ', num_frames)
    print('N          = ', N)
    print('N_avg      = ', N_avg)
    print(' ')

    # Compute velocities
    v = sa.compute_velocities(x, dt=dt_sample)

    # Compute velocity histograms
    name = file_prefix + '.' + str(i) +  '.' + str(second_index) + '.base.histogram_velocity'
    h = sa.compute_histogram_from_frames(x, v, column_sample=2, column_value=0, num_intervales=40, xmin=0, xmax=wall_Lz, N_avg=N_avg, file_prefix=name)

    # Compute velocity slope
    vel_slope_v1, vel_slope_error_v1 = sa.compute_velocity_slope(h[-1])
    vel_slope_v2, vel_slope_error_v2 = sa.compute_velocity_slope(h[-2])
    vel_slope_v3, vel_slope_error_v3 = sa.compute_velocity_slope(h[-3])
    vel_slope_v4, vel_slope_error_v4 = sa.compute_velocity_slope(h[-4])

    # Compute average velocity slope
    vel_slope_mean = ( vel_slope_v1 +  vel_slope_v2 +  vel_slope_v3) / 3.0
    vel_slope_std = np.sqrt(((vel_slope_v1 - vel_slope_mean)**2 + (vel_slope_v2 - vel_slope_mean)**2 + (vel_slope_v3 - vel_slope_mean)**2) / (3 * 2))
    vel_slope_std += (vel_slope_error_v1 + vel_slope_error_v2 + vel_slope_error_v3) / 3.0
    print('slope_v1 = ', vel_slope_v1, ' +/- ', vel_slope_error_v1)
    print('slope_v2 = ', vel_slope_v2, ' +/- ', vel_slope_error_v2)
    print('slope_v3 = ', vel_slope_v3, ' +/- ', vel_slope_error_v3)
    print('slope_v4 = ', vel_slope_v4, ' +/- ', vel_slope_error_v4)
    print('slope    = ', vel_slope_mean, ' +/- ', vel_slope_std)

    # Compute viscosity
    eta_mean = gamma_dot / vel_slope_mean
    eta_std = eta_mean * np.abs(vel_slope_std / vel_slope_mean)
    print('eta    = ', eta_mean, ' +/- ', eta_std, '\n')
    
    if True:
      # Compute viscosity from three last histograms
      eta_v1, eta_error_v1 = sa.compute_viscosity_from_profile(h[-1], gamma_dot=gamma_dot, eta_0=eta)
      eta_v2, eta_error_v2 = sa.compute_viscosity_from_profile(h[-2], gamma_dot=gamma_dot, eta_0=eta)
      eta_v3, eta_error_v3 = sa.compute_viscosity_from_profile(h[-3], gamma_dot=gamma_dot, eta_0=eta)
      eta_v4, eta_error_v4 = sa.compute_viscosity_from_profile(h[-4], gamma_dot=gamma_dot, eta_0=eta)

      # Compute average viscosity and error
      eta_mean_version_2 = (eta_v1 + eta_v2 + eta_v3) / 3.0
      eta_std_version_2 = np.sqrt(((eta_v1 - eta_mean)**2 + (eta_v2 - eta_mean)**2 + (eta_v3 - eta_mean)**2) / (3 * 2)) 
      eta_std_version_2 += (eta_error_v1 + eta_error_v2 + eta_error_v3) / 3.0
      print('eta_v1 = ', eta_v1, ' +/- ', eta_error_v1)
      print('eta_v2 = ', eta_v2, ' +/- ', eta_error_v2)
      print('eta_v3 = ', eta_v3, ' +/- ', eta_error_v3)
      print('eta_v4 = ', eta_v4, ' +/- ', eta_error_v4)
      print('eta    = ', eta_mean_version_2, ' +/- ', eta_std_version_2, '\n')

    # Compute shear rate
    shear_rate = gamma_dot / eta_mean
    shear_rate_std = shear_rate * eta_std / eta_mean

    # Average stresslet files
    force_moment_avg = np.zeros((3,3))
    force_moment_std_avg = np.zeros((3,3))
    force_moment_std = np.zeros((3,3))
    force_moment_counter = 0
    for j in range(N_samples):
      name_force_moment = file_prefix + '.' + str(i) + '.' + str(second_index) + '.' + str(j) + '.force_moment.dat'
      name_force_moment_std = file_prefix + '.' + str(i) + '.' + str(second_index) + '.' + str(j) + '.force_moment_standard_error.dat'
      try:
        f = np.loadtxt(name_force_moment)
        f_std = np.loadtxt(name_force_moment_std)
        if f_std.size > 0:
          force_moment_std += force_moment_counter * (f - force_moment_avg)**2 / (force_moment_counter + 1)
          force_moment_avg += (f - force_moment_avg) / (force_moment_counter + 1)
          force_moment_std_avg += (f_std - force_moment_std_avg) / (force_moment_counter + 1)
          force_moment_counter += 1
      except OSError:
        pass

    # Compute standard error
    force_moment_std = np.sqrt(force_moment_std / (force_moment_counter * np.maximum(1, force_moment_counter - 1))) + force_moment_std_avg

    # Compute stresslet and rotlet
    S = 0.5 * (force_moment_avg + force_moment_avg.T)
    R = 0.5 * (force_moment_avg - force_moment_avg.T)
    S_std_error = 0.5 * (force_moment_std + force_moment_std.T)
    R_std_error = 0.5 * (force_moment_std + force_moment_std.T)

    # Compute viscosity (assuming background flow = shear * (z, 0, 0)
    eta_stresslet_mean = eta + number_density * force_moment_avg[0,2] / shear_rate
    eta_stresslet_std_error = abs(eta_stresslet_mean - eta) * shear_rate_std / shear_rate + number_density * force_moment_std[0,2] / shear_rate
    print('eta    = ', eta_stresslet_mean / eta, ' +/- ', eta_stresslet_std_error / eta)

    # Magnitude norms
    rotlet_norm = np.linalg.norm(R)
    print('|S|     = ', np.linalg.norm(S))
    print('|R|     = ', rotlet_norm)
    print('|R| / D = ', rotlet_norm / np.linalg.norm(force_moment_avg))
    print('\n')

    # Store viscosity
    eta_files[k,1] = i
    eta_files[k,2] = N
    eta_files[k,3] = gamma_dot
    eta_files[k,4] = shear_rate
    eta_files[k,5] = shear_rate_std
    eta_files[k,6] = eta_mean * eta
    eta_files[k,7] = eta_std * eta
    eta_files[k,7] = eta_std * eta
    eta_files[k,8] = eta_stresslet_mean
    eta_files[k,9] = eta_stresslet_std_error
    
  # Save visocity
  name = file_prefix + '.' + str(indices[0]) + '-' + str(indices[-1]) + '.' + str(second_index) + '.base.viscosities.dat'
  np.savetxt(name, eta_files, header='Columns (10): simulation number, index, number particles, gamma_dot, shear_rate (measured), \nshear_rate_std, viscosity (from velocity profile), standard error, viscosity (from stresslet), standard error')

  
