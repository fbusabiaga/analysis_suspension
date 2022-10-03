import numpy as np
import sys
import simulation_analysis as sa


if __name__ == '__main__':
  # Set parameters
  file_prefix = '/workspace/scratch/users/fbalboa/simulations/RigidMultiblobsWall/rheology/data/run2000/run2130/run2130'
  name_vertex = '/workspace/scratch/users/fbalboa/sfw/RigidMultiblobsWall/multi_bodies/examples/rheology/Structures/star_N_13_a_0.05.vertex'
  second_index = 0
  indices = np.arange(0, 7, dtype=int)
  N_skip_stresslet = 0
  N_skip = 1
  N_hist = 4
  number_simulation = 2130
  N_samples = 10
  rcut = 0.15
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
    blob_radius =  float(read.get('blob_radius'))
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
      name = file_prefix + '.' + str(i) + '.' + str(second_index) + '.base.clusters.rcut.' + str(rcut) + '.dat' 
      clusters = np.loadtxt(name)
  
      # Cluster size histogram
      num_colloids_in_cluster = []
      cluster_length = []
      
      # Loop over frames
      for j, cj in enumerate(clusters[N_avg:]):
        for l in range(cj.size):
          index = cj[l]
          if index == l:
            colloids_indexes = np.argwhere(cj == index).flatten()
            num_colloids_in_cluster.append(colloids_indexes.size)

            # # Get bodies coordinates
            # r_bodies = np.zeros((colloids_indexes.size, 3))
            # for m, ind in enumerate(colloids_indexes):
            #   r_bodies[m] = x[j, ind, 0:3]
          
            # # Compute vector distance
            # rx = r_bodies[:,0]
            # ry = r_bodies[:,1]
            # rz = r_bodies[:,2]
            # dx = (rx[:,None] - rx).flatten()
            # dy = (ry[:,None] - ry).flatten()
            # dz = (rz[:,None] - rz).flatten()

            # # Project to PBC
            # if L[0] > 0:
            #   sel_p = dx > 0
            #   sel_n = dx < 0
            #   dx[sel_p] = dx[sel_p] - (dx[sel_p] / L[0] + 0.5).astype(int) * L[0]
            #   dx[sel_n] = dx[sel_n] - (dx[sel_n] / L[0] - 0.5).astype(int) * L[0]
            # if L[1] > 0:
            #   sel_p = dy > 0
            #   sel_n = dy < 0
            #   dy[sel_p] = dy[sel_p] - (dy[sel_p] / L[1] + 0.5).astype(int) * L[1]
            #   dy[sel_n] = dy[sel_n] - (dy[sel_n] / L[1] - 0.5).astype(int) * L[1]
            # if L[2] > 0:
            #   sel_p = dz > 0
            #   sel_n = dz < 0
            #   dz[sel_p] = dz[sel_p] - (dz[sel_p] / L[2] + 0.5).astype(int) * L[2]
            #   dz[sel_n] = dz[sel_n] - (dz[sel_n] / L[2] - 0.5).astype(int) * L[2]

            # # Get maximum length
            # dr = np.sqrt(dx**2 + dy**2 + dz**2).flatten()
            # cluster_length.append(np.max(dr))
          if index == -1:
            num_colloids_in_cluster.append(1)
            # cluster_length.append(0)

      # Transform to numpy array
      num_colloids_in_cluster = np.array(num_colloids_in_cluster, dtype=int).flatten()
      # cluster_length = np.array(cluster_length).flatten() 
      print('num_colloids_in_cluster = ', num_colloids_in_cluster.shape)
      print('num_colloids_in_cluster = ', num_colloids_in_cluster)

      # Cluster size
      name = file_prefix + '.' + str(i) + '.' + str(second_index) + '.base.histogram.cluster_size.rcut.' + str(rcut) + '.dat'
      sort = np.sort(num_colloids_in_cluster)
      xmin = -0.5
      xmax = 2*sort[-1]-sort[-2]
      # max_points_bin = int(np.sqrt(num_colloids_in_cluster.size))+1
      max_points_bin = int(np.log2(num_colloids_in_cluster.size))+1
      max_levels = np.log2(xmax - xmin) - 1
      print('Histogram N')
      print('xmax           = ', xmax)
      print('xmin           = ', xmin)
      print('max_points_bin = ', max_points_bin)
      print('max_levels     = ', max_levels)
      sa.compute_histogram(num_colloids_in_cluster, xmin=xmin, xmax=xmax,
                           scale='adaptive', max_points_bin=max_points_bin, max_levels=max_levels,
                           name=name)

      # Cluster length
      # name = file_prefix + '.' + str(i) + '.' + str(second_index) + '.base.histogram.cluster_length.rcut.' + str(rcut) + '.dat'
      # sort = np.sort(cluster_length)
      # xmin = 0
      # xmax = 2*sort[-1]-sort[-2]
      # max_points_bin = int(np.log2(cluster_length.size))+1
      # max_levels = np.log2((xmax - xmin) / blob_radius) - 2
      # sa.compute_histogram(cluster_length, xmin=xmin, xmax=xmax,
      #                      scale='adaptive', max_points_bin=max_points_bin, max_levels=max_levels,
      #                      name=name)

      # Cluster averages
      fraction_in_cluster_mean = 0
      fraction_in_cluster_error = 0
      # Loop over frame blocks
      for j in range(N_skip, N_hist):
        cj = clusters[j*N_avg : (j+1)*N_avg]
        sel = cj.flatten() > -1
        fraction_in_cluster = np.sum(sel) / sel.size
        fraction_in_cluster_error += (j - N_skip) * (fraction_in_cluster - fraction_in_cluster_mean)**2 / (j - N_skip + 1)
        fraction_in_cluster_mean += (fraction_in_cluster - fraction_in_cluster_mean) / (j - N_skip + 1)
        print('j, fraction_in_cluster = ', j, fraction_in_cluster)
      fraction_in_cluster_error = np.sqrt(fraction_in_cluster_error / np.maximum(1, N_hist-N_skip) / np.maximum(1, N_hist-N_skip-1))
      print('fraction_in_cluster = ', fraction_in_cluster_mean, ' +/- ', fraction_in_cluster_error)

      # Cluster mean size
      Nc = np.sum(num_colloids_in_cluster) / num_colloids_in_cluster.size
      sel = num_colloids_in_cluster > 1
      Nc_cluster = np.sum(num_colloids_in_cluster[sel]) / num_colloids_in_cluster[sel].size
      print('Nc = ', Nc, Nc_cluster)
      
      # Cluster mean length
      # Lc = np.sum(cluster_length) / cluster_length.size
      # sel = cluster_length > 0
      # Lc_cluster = np.sum(cluster_length[sel]) / cluster_length[sel].size
      # print('Lc = ', Lc, Lc_cluster)
      
      # Save info
      cluster_files[k,1] = i
      cluster_files[k,2] = N
      cluster_files[k,3] = gamma_dot
      cluster_files[k,4] = fraction_in_cluster_mean
      cluster_files[k,5] = fraction_in_cluster_error
      cluster_files[k,6] = Nc
      cluster_files[k,7] = Nc_cluster
      #cluster_files[k,8] = Lc
      #cluster_files[k,9] = Lc_cluster

   
    
  if detect_clusters:
    # Save clusters info
    name = file_prefix + '.' + str(indices[0]) + '-' + str(indices[-1]) + '.' + str(second_index) + '.base.clusters.rcut.' + str(rcut) + '.dat'
    header = 'Columns (10): 0=simulation number, 1=index, 2=number particles, 3=gamma_dot, 4=cluster size, 5=cluster size error, 6=Nc, 7=Nc_cluster, 8=Lc, 9=Lc_cluster'
    np.savetxt(name, cluster_files, header=header)
