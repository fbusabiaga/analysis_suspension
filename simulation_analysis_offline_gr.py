import numpy as np
import sys
import simulation_analysis as sa


if __name__ == '__main__':
  # Set parameters
  file_prefix = '/workspace/scratch/users/fbalboa/simulations/RigidMultiblobsWall/rheology/data/run2000/run2113/run2113'
  name_vertex = '/workspace/scratch/users/fbalboa/sfw/RigidMultiblobsWall/multi_bodies/examples/rheology/Structures/star_hook_N_25_a_0.05.vertex'
  num_frames = 2000
  num_frames_skip = 20
  rcut = 0.3
  nbins = 100
  dim = '3d'
  gr_type = 'blobs' # 'bodies' or 'blobs'
  second_index = 0
  indices = np.arange(0, 7, dtype=int)
  number_simulation = 2113
  N_samples = 1
  print('indices = ', indices)

  # Prepare viscosity file
  eta_files = np.zeros((len(indices), 11))
  eta_files[:,0] = number_simulation
  
  # Loop over files
  for k, i in enumerate(indices):  
    name_input = file_prefix + '.' + str(i) + '.' + str(second_index) + '.0.inputfile'
    print('name_input  = ', name_input)        

    # Read inputfile
    read = sa.read_input(name_input)

    # Read vertex
    r_vectors = sa.read_vertex(name_vertex)

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
    print('x.shape    = ', x.shape)
    print('num_frames = ', num_frames)
    print('N          = ', N)
    print(' ')

    # Call gr
    name = file_prefix + '.' + str(i) + '.' + str(second_index) + '.base.gr.' + gr_type + '.dat'
    L_gr = np.array([L[0], L[1], 10 * (L[0] + L[1])])
    if gr_type == 'bodies':
      r_vectors_gr = None
    elif gr_type == 'blobs':
      r_vectors_gr = r_vectors
    gr = sa.radial_distribution_function(x[num_frames_skip:], num_frames, rcut=rcut, nbins=nbins, r_vectors=r_vectors_gr, L=L_gr, dim=dim, name=name)

