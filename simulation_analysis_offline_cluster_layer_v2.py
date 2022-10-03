import numpy as np
import sys
import simulation_analysis as sa


if __name__ == '__main__':
  # Set parameters
  file_prefix = '/home/fbalboa/simulations/RigidMultiblobsWall/rheology/data/run2000/run2123/run2123'
  # name_vertex = '/home/fbalboa/sfw/RigidMultiblobsWall/multi_bodies/examples/rheology/Structures/star_N_13_a_0.05.vertex'
  name_vertex = '/home/fbalboa/sfw/RigidMultiblobsWall/multi_bodies/examples/rheology/Structures/star_hook_N_33_a_0.05.vertex'
  second_index = 0
  indices = np.arange(0, 7, dtype=int)
  N_hist = 4
  number_simulation = 2123
  N_samples = 1
  rcut = 0.1
  detect_clusters = True
  print('indices = ', indices)

  # Prepare viscosity file
  cluster_files = np.zeros((len(indices), 10))
  cluster_files[:,0] = number_simulation

  # Read vertex
  r_vectors = sa.read_vertex(name_vertex)
  
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
    names_config = []
    for j in range(N_samples):
      names_config.append(file_prefix + '.' + str(i) + '.' + str(second_index) + '.' + str(j) + '.star_run' + str(number_simulation) + '.' + str(i) + '.' + str(second_index) + '.' + str(j) + '.config')
    x = sa.read_config_list(names_config, print_name=True)
    num_frames = x.shape[0]
    num_frames_vel = num_frames - 1
    N_avg = (num_frames-1) // N_hist
    print('x.shape    = ', x.shape)
    print('num_frames = ', num_frames)
    print('N          = ', N)
    print('N_avg      = ', N_avg)
    print(' ')
    
    # Detect clusters
    if detect_clusters:
      name = file_prefix + '.' + str(i) + '.' + str(second_index) + '.base.clusters.dat' 
      clusters = np.loadtxt(name)
  
      # Cluster size histogram
      num_colloids_in_cluster = []
      cluster_length = []
      
      # Loop over frames
      for j, cj in enumerate(clusters[N_avg:num_frames]):
        for l in range(cj.size):
          index = cj[l]
          if index == l:
            colloids_indexes = np.argwhere(cj == index).flatten()
            num_colloids_in_cluster.append(colloids_indexes.size)

            # Get bodies coordinates
            r_bodies = np.zeros((colloids_indexes.size, 3))
            for m, ind in enumerate(colloids_indexes):
              r_bodies[m] = x[j, ind, 0:3]
          
            # Compute vector distance
            rx = r_bodies[:,0]
            ry = r_bodies[:,1]
            rz = r_bodies[:,2]
            dx = (rx[:,None] - rx).flatten()
            dy = (ry[:,None] - ry).flatten()
            dz = (rz[:,None] - rz).flatten()

            # Project to PBC
            if L[0] > 0:
              sel_p = dx > 0
              sel_n = dx < 0
              dx[sel_p] = dx[sel_p] - (dx[sel_p] / L[0] + 0.5).astype(int) * L[0]
              dx[sel_n] = dx[sel_n] - (dx[sel_n] / L[0] - 0.5).astype(int) * L[0]
            if L[1] > 0:
              sel_p = dy > 0
              sel_n = dy < 0
              dy[sel_p] = dy[sel_p] - (dy[sel_p] / L[1] + 0.5).astype(int) * L[1]
              dy[sel_n] = dy[sel_n] - (dy[sel_n] / L[1] - 0.5).astype(int) * L[1]
            if L[2] > 0:
              sel_p = dz > 0
              sel_n = dz < 0
              dz[sel_p] = dz[sel_p] - (dz[sel_p] / L[2] + 0.5).astype(int) * L[2]
              dz[sel_n] = dz[sel_n] - (dz[sel_n] / L[2] - 0.5).astype(int) * L[2]

            # Get maximum length
            dr = np.sqrt(dx**2 + dy**2 + dz**2).flatten()
            cluster_length.append(np.max(dr))
          if index == -1:
            num_colloids_in_cluster.append(1)
            cluster_length.append(0)

      # Transform to numpy array
      num_colloids_in_cluster = np.array(num_colloids_in_cluster, dtype=int).flatten()
      cluster_length = np.array(cluster_length).flatten() 
      name = file_prefix + '.' + str(i) + '.' + str(second_index) + '.base.histogram.cluster_size.adaptive.dat'
      print('num_colloids_in_cluster = ', num_colloids_in_cluster.shape)
      print('num_colloids_in_cluster = ', num_colloids_in_cluster)
      # num_intervales=int(np.log(np.max(num_colloids_in_cluster)+1))
      num_intervales = np.array([0.99, 1.99, 2.99, 3.99, np.max(num_colloids_in_cluster)+1])
      xmin=0.99
      xmax=np.max(num_colloids_in_cluster)+1
      max_points_bin=(np.sum(num_colloids_in_cluster))**0.5 / 5
      print('num_intervales = ', num_intervales)
      print('xmin           = ', xmin)
      print('xmax           = ', xmax)
      print('max_points_bin = ', max_points_bin)
      
      sa.compute_histogram(num_colloids_in_cluster,
                           scale='adaptive',
                           num_intervales=num_intervales,
                           xmin=xmin,
                           xmax=xmax,
                           max_points_bin=max_points_bin,
                           name=name)

      

   
