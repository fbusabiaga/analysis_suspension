'''
Script to save escape times for a two particle cluster.
'''
import numpy as np
import sys
import simulation_analysis as sa


if __name__ == '__main__':
  # Set parameters
  file_prefix = '/workspace/scratch/users/fbalboa/simulations/RigidMultiblobsWall/rheology/data/run2000/run2022/run2022.3.0'
  N_start = 0
  N_end = 400
  N = N_end - N_start
  N_avg = 4

  # Prepare output, columns: number of frames
  escape_times = np.zeros(N)

  # Loop over simulations
  for i in range(N_start, N_end):    
    # Read inputfile
    name_input = file_prefix + '.' + str(i) + '.inputfile' 
    read = sa.read_input(name_input)
    dt = float(read.get('dt')) 
    n_save = int(read.get('n_save'))
    dt_sample = dt * n_save
    gamma_dot = float(read.get('gamma_dot'))

  
    # Read number of steps
    name = file_prefix + '.' + str(i) + '.number_of_steps.dat'
    with open(name) as f_handle:
      steps = int(f_handle.readline())
    escape_times[i - N_start] = steps * dt
    print('i = ', i, ', escape_time = ', steps * dt)
    
  # Save result 
  name = file_prefix + '.' + str(N_start) + '-' + str(N_end-1) + '.escape_times.dat'
  np.savetxt(name, escape_times)

  # Prepare variables
  N_hist = N // N_avg
  num_intervales = int(np.sqrt(N))
  hist = np.zeros((N_avg, num_intervales, 3))
  hist_log = np.zeros((N_avg, num_intervales, 3))
  xmin = np.min(escape_times) / 1.25
  xmax = np.max(escape_times) * 1.25
  xmin_log = np.log10(gamma_dot * np.min(escape_times) / 2)
  xmax_log = np.log10(gamma_dot * np.max(escape_times) * 2)

  for i in range(N_avg):
    # Hist linear in time
    h = sa.compute_histogram(escape_times[i*N_hist:(i+1)*N_hist].reshape((N_hist,1)), num_intervales=num_intervales, xmin=0, xmax=xmax)
    hist[i] = np.copy(h)
    
    # Hist log in time
    h = sa.compute_histogram(np.log10(gamma_dot * escape_times[i*N_hist:(i+1)*N_hist]).reshape((N_hist, 1)), 
                             num_intervales=num_intervales, xmin=xmin_log, xmax=xmax_log)
    hist_log[i] = np.copy(h)
    
  # Save files
  h = np.zeros((num_intervales, 5))
  h[:,0] = hist[0,:,0]
  h[:,1:3] = np.average(hist[:,:,1:3], axis=0)
  h[:,3:5] = np.std(hist[:,:,1:3], axis=0, ddof=1) / np.sqrt(N_avg)
  name = file_prefix + '.' + str(N_start) + '-' + str(N_end-1) + '.histogram.escape_times.dat'
  np.savetxt(name, h, header='Columns: t, density, number, std density, std number')
  
  h = np.zeros((num_intervales, 5))
  h[:,0] = hist_log[0,:,0]
  h[:,1:3] = np.average(hist_log[:,:,1:3], axis=0)
  h[:,3:5] = np.std(hist_log[:,:,1:3], axis=0, ddof=1) / np.sqrt(N_avg)
  name = file_prefix + '.' + str(N_start) + '-' + str(N_end-1) + '.histogram.escape_times_log.dat'
  np.savetxt(name, h, header='Columns: log10(gamma_dot*t), density, number, std density, std number')

