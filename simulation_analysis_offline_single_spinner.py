'''
Script to analyze a single spinner.
'''
import numpy as np
import sys
import simulation_analysis as sa


if __name__ == '__main__':
  # Set parameters
  file_prefix = '/home/fbalboa/simulations/RigidMultiblobsWall/chiral/data/run2000/run2103/run2103'
  indices = np.arange(20, 27, dtype=int)
  force_pull = np.array([0, 0.01, 0.05, 0.1, 0.5, 1, 5])
  # force_pull = np.array([0, 0.1, 1])
  frame_rate = 1
  
  print('indices = ', indices)

  # Create arrays for variables
  popt_x = np.zeros((force_pull.size, 5))
  pcov_x = np.zeros((force_pull.size, 5, 5))
  R2_x = np.zeros(force_pull.size)
  popt_y = np.zeros((force_pull.size, 5))
  pcov_y = np.zeros((force_pull.size, 5, 5))
  R2_y = np.zeros(force_pull.size)
  arccos_omega_z = np.zeros((force_pull.size, 2))
  
  # Loop over files
  for k, i in enumerate(indices):
    name_input = file_prefix + '.' + str(i) + '.0.0.inputfile' 
    print('name_input  = ', name_input) 
    
    # Read inputfile
    read = sa.read_input(name_input)

    # Get number of particles
    name_config = file_prefix + '.' + str(i) + '.0.0.shell.config'
    N = sa.read_particle_number(name_config)

    # Set some parameters
    dt = float(read.get('dt')) 
    n_save = int(read.get('n_save'))
    dt_sample = dt * n_save
    eta = float(read.get('eta'))
    omega = float(read.get('omega'))
    print('dt        = ', dt)
    print('dt_sample = ', dt_sample)
    print('n_save    = ', n_save)
    print('eta       = ', eta)
    print('omega     = ', omega)
    print(' ')

    # Read config
    name_config = file_prefix + '.' + str(i) + '.0.0.shell.config'
    print('name_config = ', name_config)
    x = sa.read_config(name_config)
    t = np.arange(x.shape[0]) * dt
    
    num_frames = x.shape[0]
    num_frames_vel = num_frames - frame_rate
    print('x.shape    = ', x.shape)
    print('num_frames = ', num_frames)
    print('N          = ', N)
    print(' ')

    # Fit x position
    def func(t, x0, slope, amplitude, omega, phi):
      return x0 + slope * t + amplitude * np.sin(omega * t + phi)
    p0 = [0.01, 0.01, 0.01, omega, 0.01]
    name = file_prefix + '.' + str(i) + '.0.0.fit_x.png'
    popt_x_i, pcov_x_i, R2_x_i = sa.nonlinear_fit(t, x[:,0,0], func, sigma=None, p0=p0, save_plot_name=name)
    popt_x[k,:] = popt_x_i
    pcov_x[k,:,:] = pcov_x_i
    R2_x[k] = R2_x_i
    
    # Fit y position
    def func(t, x0, slope, amplitude, omega, phi):
      return x0 + slope * t + amplitude * np.sin(omega * t + phi)
    p0 = [0.01, 0.01, 0.01, omega, 0.01]
    name = file_prefix + '.' + str(i) + '.0.0.fit_y.png'
    popt_y_i, pcov_y_i, R2_y_i = sa.nonlinear_fit(t, x[:,0,1], func, sigma=None, p0=p0, save_plot_name=name)
    popt_y[k,:] = popt_y_i
    pcov_y[k,:,:] = pcov_y_i
    R2_y[k] = R2_y_i

    # Compute velocity
    velocity = sa.compute_velocities(x, dt=dt, frame_rate=frame_rate)
    print('velocity = ', velocity.shape)
    velocity = velocity.reshape((num_frames_vel, 6))
    print('velocity = ', velocity.shape)

    # Compute angle between angular velocity and vertical
    z = np.array([0, 0, 1])
    theta = np.arccos(np.einsum('ik,k->i', velocity[:,3:], z) / np.linalg.norm(velocity[:,3:], axis=1)) * (180 / np.pi)
    arccos_omega_z[k,0] = np.mean(theta)
    arccos_omega_z[k,1] = np.std(theta, ddof=1)

    # Save angle with vertical
    name = file_prefix + '.' + str(i) + '.0.0.arccos_omega_z_vs_time.dat'
    result = np.zeros((theta.size, 2))
    result[:,0] = t[:-1]
    result[:,1] = theta
    np.savetxt(name, result, header='Columns: time, angle with z (degrees)')

    # End of loop
    print('==========\n\n\n')

    
  # Save fits
  header = 'Columns: force_pull, y_slope, y_slope_error, y_R2, y_slope, y_slope_error, y_R2'
  result = np.zeros((force_pull.size, 7))
  result[:,0] = force_pull
  result[:,1] = popt_x[:,1]
  result[:,2] = np.sqrt(pcov_x[:,1,1])
  result[:,3] = R2_x
  result[:,4] = popt_y[:,1]
  result[:,5] = np.sqrt(pcov_y[:,1,1])
  result[:,6] = R2_y
  name = file_prefix + '.' + str(indices[0]) + '-' + str(indices[-1]) + '.slope_fits.dat'
  np.savetxt(name, result, header=header)

  # Save angle with vertical
  header = 'Columns: force_pull, angle with vertical (degrees), standard deviation'
  result = np.zeros((force_pull.size, 3))
  result[:,0] = force_pull
  result[:,1:] = arccos_omega_z
  name = file_prefix + '.' + str(indices[0]) + '-' + str(indices[-1]) + '.arccos_omega_z.dat'
  np.savetxt(name, result, header=header)
