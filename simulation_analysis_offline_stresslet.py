import numpy as np
import sys
import simulation_analysis as sa


if __name__ == '__main__':
  # Set parameters
  file_prefix = '/home/fbalboa/simulations/RigidMultiblobsWall/rheology/data/run2000/run2002/run2002'
  second_index = 0
  indices = np.arange(17, 18, dtype=int)
  number_simulation = 2002
  N_samples = 1
  print('indices = ', indices)

  # Prepare viscosity file
  eta_files = np.zeros((len(indices), 6))
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
    print('dt             = ', dt)
    print('dt_sample      = ', dt_sample)
    print('n_save         = ', n_save)
    print('eta            = ', eta)
    print('gamme_dot      = ', gamma_dot)
    print('L              = ', L)
    print('wall_Lz        = ', wall_Lz)
    print('volume         = ', volume)
    print('number_density = ', number_density)
    print(' ')
  
    # Average stresslet files
    force_moment_avg = np.zeros((3,3))
    force_moment_std_avg = np.zeros((3,3))
    force_moment_std = np.zeros((3,3))
    force_moment_counter = 0
    for j in range(N_samples):
      name_force_moment = file_prefix + '.' + str(i) + '.' + str(second_index) + '.' + str(j) + '.force_moment.dat'
      name_force_moment_std = file_prefix + '.' + str(i) + '.' + str(second_index) + '.' + str(j) + '.force_moment_standard_error.dat'
      print('name_config = ', name_config)
      try:
        f = np.loadtxt(name_force_moment)
        f_std = np.loadtxt(name_force_moment_std)
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

    # Magnitude norms
    rotlet_norm = np.linalg.norm(R)
    print('|S|     = ', np.linalg.norm(S))
    print('|R|     = ', rotlet_norm)
    print('|R| / D = ', rotlet_norm / np.linalg.norm(force_moment_avg))
    print('\n')

    # Compute viscosity (assuming background flow = shear * (z, 0, 0)
    eta_mean = number_density * force_moment_avg[0,2] / gamma_dot 
    eta_std_error = number_density * force_moment_std[0,2] / gamma_dot 
    print('eta = ', eta_mean / eta, ' +/- ', eta_std_error / eta)
    print('\n')

    # Store viscosity
    eta_files[k,1] = i
    eta_files[k,2] = N
    eta_files[k,3] = gamma_dot
    eta_files[k,4] = eta_mean + eta
    eta_files[k,5] = eta_std_error

    # Save visocity
  name = file_prefix + '.' + str(indices[0]) + '-' + str(indices[-1]) + '.' + str(second_index) + '.base.viscosities_from_stresslet.dat'
  np.savetxt(name, eta_files, header='Viscosity measured from the force moment\nColumns: simulation number, index, number particles, gamma_dot, viscosity, standard error')

