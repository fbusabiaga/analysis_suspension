import numpy as np
import sys
import simulation_analysis as sa


if __name__ == '__main__':
  # Set parameters
  file_prefix = '/workspace/scratch/users/fbalboa/simulations/RigidMultiblobsWall/rheology/data/run2000/run2110/run2110'
  second_index = 0
  indices = np.arange(0, 7, dtype=int)
  N_skip_stresslet = 0
  N_skip = 1
  N_hist = 4
  number_simulation = 2110
  N_samples = 3
  print('indices = ', indices)

  # Prepare viscosity file
  eta_files = np.zeros((len(indices), 13))
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

    # Compute velocity slopes
    vel_slope_mean = 0
    vel_slope_std = 0
    vel_slope_mean_std = 0
    for j in range(N_hist):
      vel_slope, vel_slope_error = sa.compute_velocity_slope(h[j])
      print('j = ', j, ' slope = ', vel_slope, ' +/- ', vel_slope_error)

      # Compute means
      if j >= N_skip:
        vel_slope_mean_std += (j - N_skip) * (vel_slope - vel_slope_mean)**2 / (j - N_skip + 1)
        vel_slope_mean += (vel_slope - vel_slope_mean) / (j - N_skip + 1)
        vel_slope_std += (vel_slope_error - vel_slope_std) / (j - N_skip + 1)

    vel_slope_std += np.sqrt(vel_slope_mean_std / np.maximum(1, N_hist - N_skip) / np.maximum(1, N_hist - N_skip - 1))
    print('slope  = ', vel_slope_mean, ' +/- ', vel_slope_std)

    # Compute viscosity from slopes average
    eta_mean = gamma_dot / vel_slope_mean
    eta_std = np.abs(eta_mean * vel_slope_std / vel_slope_mean)
    print('eta    = ', eta_mean, ' +/- ', eta_std, '\n')
    
    # Compute viscosities from slopes and then average
    if True:
      eta_v_mean = 0
      eta_v_std = 0
      eta_v_mean_std =0
      for j in range(N_hist):
        eta_v, eta_v_error = sa.compute_viscosity_from_profile(h[j], gamma_dot=gamma_dot, eta_0=eta)
        print('j = ', j, ' eta_v = ', eta_v, ' +/- ', eta_v_error)

        # Compute means
        if j >= N_skip:
          eta_v_mean_std += (j - N_skip) * (eta_v - eta_v_mean)**2 / (j - N_skip + 1)
          eta_v_mean += (eta_v - eta_v_mean) / (j - N_skip + 1)
          eta_v_std += (eta_v_error - eta_v_std) / (j - N_skip + 1)
          
      eta_v_std += np.sqrt(eta_v_mean_std / np.maximum(1, N_hist - N_skip) / np.maximum(1, N_hist - N_skip - 1))
      print('eta = ', eta_v_mean, ' +/- ', eta_v_std, '\n\n')

    # Compute shear rate
    shear_rate = gamma_dot / eta_mean
    shear_rate_std = shear_rate * eta_std / eta_mean

    # Average stresslet files
    force_moment_avg = np.zeros((3,3))
    force_moment_std_avg = np.zeros((3,3))
    force_moment_std = np.zeros((3,3))
    force_moment_counter = 0
    for j in range(N_skip_stresslet, N_samples):
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

    D_mean = 0
    D_error = 0
    D_error_mean = 0
    for j in range(N_hist):
      name = file_prefix + '.' + str(i) + '.' + str(second_index) + '.base.msd.' + str(j).zfill(4) + '.dat' 
      msd, msd_std = sa.msd(x[N_avg * j : N_avg * (j+1)], dt_sample, MSD_steps=num_frames, output_name=name)
      D = msd[1,4] / (2 * dt_sample)
      if j >= N_skip:
        D_error += (j - N_skip) * (D - D_mean)**2 / (j - N_skip + 1)
        D_mean += (D - D_mean) / (j - N_skip + 1)
      print('D = ', D)
    D_error = np.sqrt(D_error) / np.maximum(1, N_hist - N_skip) / np.maximum(1, N_hist - N_skip - 1))
    print('D = ', D_mean, ' +/-', D_error)
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
    eta_files[k,10] = num_frames * dt_sample
    eta_files[k,11] = D_mean
    eta_files[k,12] = D_error
    
  # Save visocity
  name = file_prefix + '.' + str(indices[0]) + '-' + str(indices[-1]) + '.' + str(second_index) + '.base.viscosities.dat'
  np.savetxt(name, eta_files, header='Columns (11): 0=simulation number, 1=index, 2=number particles, 3=gamma_dot, 4=shear_rate (measured), \n5=shear_rate_std, 6=viscosity (from velocity profile), 7=standard error, 8=viscosity (from stresslet), 9=standard error, 10=time, 11=D(yy,t=dt_sample), 12=D_error(yy,t=dt_sample)')

  
