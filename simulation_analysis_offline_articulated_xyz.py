'''
Script to save xyz files.
'''
import numpy as np
import sys
import simulation_analysis as sa


if __name__ == '__main__':
  # Set parameters
  file_prefix = '/home/fbalboa/simulations/RigidMultiblobsWall/articulated/data/run0/run104/run104.0.3.0'
  files_config = ['/home/fbalboa/simulations/RigidMultiblobsWall/articulated/data/run0/run104/run104.0.3.0.bacteria.config']
  list_vertex = '/home/fbalboa/simulations/RigidMultiblobsWall/articulated/data/run0/run104/run104.0.3.0.bacteria_run104.0.0.0.list_vertex'
  folder_vertex = '/home/fbalboa/sfw/RigidMultiblobsWall/multi_bodies/examples/bacteria/'
  structure = 'bacteria_frame_body'
  num_frames = 10000
  save_blobs = True
  save_tracking_points = False
  frame_body = 0
  save_dat_index = [0]

  # Set blob radius by hand
  if False:
    blob_radii = None
  elif False:
    blob_radii = []
    for i in range(200):
      if i % 2 == 0:
        blob_radii.append(np.ones(1))
      else:
        blob_radii.append(np.ones(1) * 0)    
  else:
    blob_radii = []
    for i in range(2):
      if i % 2 == 0:
        blob_radii.append(np.ones(162) * 0.131)
      else:
        blob_radii.append(np.ones(408) * 2.449489742783179935e-02)
        
    
  # Read inputfile
  name_input = file_prefix + '.inputfile' 
  read = sa.read_input(name_input)

  # Read vertex  
  r_vectors = sa.read_vertex_file_list(list_vertex, path=folder_vertex)
  print('r_vectors = ', len(r_vectors))
  for r in r_vectors:
    print('r_vectors = ', r.shape)

  # Get number of particles
  N = sa.read_particle_number(files_config[0])

  # Set some parameters
  dt = float(read.get('dt')) 
  n_save = int(read.get('n_save'))
  dt_sample = dt * n_save
  eta = float(read.get('eta'))
  periodic_length = np.fromstring(read.get('periodic_length'), sep=' ') if read.get('periodic_length') else np.zeros(3)
  print('file_prefix     = ', file_prefix)
  print('dt              = ', dt)
  print('dt_sample       = ', dt_sample)
  print('n_save          = ', n_save)
  print('eta             = ', eta)
  print('periodic_length = ', periodic_length)
  print(' ')

  # Loop over config files
  x = sa.read_config_list(files_config, print_name=True)

  # Create time array
  t = np.arange(x.shape[0]) * dt_sample
  print('total number of frames = ', x.shape[0])
  print('num_frames             = ', num_frames)
  print(' ')

  # Save xyz for blobs
  if save_blobs:   
    name = file_prefix + '.' + structure + '.xyz'
    sa.save_xyz(x, r_vectors, name, num_frames=num_frames, letter=structure[0].upper(), articulated=True, blob_vector_constant=blob_radii,
                periodic_length=periodic_length, frame_body=frame_body)

  if save_tracking_points:
    q = []
    for i in range(len(r_vectors)):
      q.append(np.zeros(3))
    name = file_prefix + '.' + structure + '.tracking_points.xyz'
    sa.save_xyz(x, q, name, num_frames=num_frames, letter=structure[0].upper(), articulated=True, blob_vector_constant=blob_radii,
                periodic_length=periodic_length)

  # Save dat file
  if save_dat_index is not None:
    for i in save_dat_index:
      name = file_prefix + '.' + structure + '.' + str(i) + '.dat'
      sa.save_dat(x, t, i, name)
