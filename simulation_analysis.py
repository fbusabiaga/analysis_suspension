import numpy as np
import scipy.stats as scst
import scipy.optimize as scop
import matplotlib.pyplot as plt
import sys


def read_input(name):
  '''
  Build a dictionary from an input file.
  The symbol # marks comments.
  '''
  
  # Comment symbol and empty dictionary
  comment_symbols = ['#']
  options = {}

  # Read input file
  with open(name, 'r') as f:
    # Loop over lines
    for line in f:
      # Strip comments
      if comment_symbols[0] in line:
        line, comment = line.split(comment_symbols[0], 1)

      # Save options to dictionary, Value may be more than one word
      line = line.strip()
      if line != '':
        option, value = line.split(None, 1)
        options[option] = value

  return options
  

def read_config(name):
  '''
  Read config and store in an array of shape (num_frames, num_bodies, 7).
  '''

  # Read number of lines and bodies
  N = 0
  try:
    with open(name, 'r') as f_handle:
      num_lines = 0
      N = 0
      for line in f_handle:
        if num_lines == 0:
          N = int(line)
        num_lines += 1
  except OSError:
    return np.array([])

  # Set array
  num_frames = num_lines // (N + 1) 
  x = np.zeros((num_frames, N, 7))

  # Read config
  with open(name, 'r') as f_handle:
    for k, line in enumerate(f_handle):
      if (k % (N+1)) == 0:
        continue
      else:
        data = np.fromstring(line, sep=' ')
        frame = k // (N+1)      
        i = k - 1 - (k // (N+1)) * (N+1)
        if frame >= num_frames:
          break
        x[frame, i] = np.copy(data)

  # Return config
  return x


def read_particle_number(name):
  '''
  Read the number of particles from the config file.
  '''

  # Read number of lines and bodies
  N = 0
  with open(name, 'r') as f_handle:
    num_lines = 0
    return int(f_handle.readline())
  return None

    
def compute_velocities(x, dt=1, frame_rate=1):
  '''
  Compute velocities between frames.
  '''
  num_frames = x.shape[0]
  N = x.shape[1]
  dt_frames = frame_rate * dt
  Psi = np.zeros((N, 4, 3))
  v = np.zeros((num_frames-frame_rate, N, 6))


  # Loop over frames and bodies
  for i in range(num_frames - frame_rate):
    # Linear velocity
    v[i,:,0:3] = (x[i+frame_rate, :, 0:3] - x[i, :, 0:3]) / dt_frames
    
    # Angular velocity
    Psi[:, 0, 0] = -x[i, :, 3 + 1]
    Psi[:, 0, 1] = -x[i, :, 3 + 2]
    Psi[:, 0, 2] = -x[i, :, 3 + 3]
    Psi[:, 1, 0] =  x[i, :, 3 + 0]
    Psi[:, 1, 1] =  x[i, :, 3 + 3]
    Psi[:, 1, 2] = -x[i, :, 3 + 2]
    Psi[:, 2, 0] = -x[i, :, 3 + 3]
    Psi[:, 2, 1] =  x[i, :, 3 + 0]
    Psi[:, 2, 2] =  x[i, :, 3 + 1]
    Psi[:, 3, 0] =  x[i, :, 3 + 2]
    Psi[:, 3, 1] = -x[i, :, 3 + 1]
    Psi[:, 3, 2] =  x[i, :, 3 + 0]
    Psi = 0.5 * Psi
    for j in range(N):
      v[i, j, 3:6] = 4.0 * np.dot(Psi[j].T, x[i + frame_rate, j, 3:7]) / dt_frames

  # Return velocities
  return v


def compute_histogram_from_frames(sample, value, column_sample=0, column_value=0, num_intervales=10, xmin=0, xmax=1, N_avg=1, file_prefix=None):
  '''
  Compute histogram averages. Use column_sample to select the bin and column_value to compute the average.
  '''
  
  # Loop over steps
  N_frames = np.minimum(sample.shape[0], value.shape[0])
  N_steps = N_frames // N_avg
  h = np.zeros((N_steps, num_intervales, 3))
  for i in range(N_steps):
    mean_avg = np.zeros(num_intervales)
    std_avg = np.zeros(num_intervales)
    counter = np.zeros(num_intervales)

    for j in range(N_avg):
      # Created binned average
      mean, bin_edges, binnumber = scst.binned_statistic_dd(sample[i*N_avg + j, :, column_sample], value[i*N_avg + j, :, column_value], bins=num_intervales, range=[[xmin, xmax]], statistic='mean')

      # Select only non empty bins
      sel = ~np.isnan(mean)      
      std_avg[sel] += counter[sel] * (mean[sel] - mean_avg[sel])**2 / (counter[sel] + 1) 
      mean_avg[sel] += (mean[sel] - mean_avg[sel]) / (counter[sel] + 1)
      counter[sel] += 1

    # Store histogram
    h[i,:,0] = 0.5 * (bin_edges[0][0:-1] + bin_edges[0][1:])
    h[i,:,1] = mean_avg
    h[i,:,2] = np.sqrt(std_avg / (np.maximum(1, counter) * np.maximum(1, counter - 1)))

    # Save to a file
    if file_prefix is not None:
      name = file_prefix + '.' + str(i).zfill(4) + '.dat'      
      np.savetxt(name, h[i], header='Columns, x, value, standard error')

  return h


def compute_viscosity_from_profile(x, gamma_dot=0, eta_0=1):
  '''
  Compute visocity from velocity profile.
  '''
  def profile(x, v0, eta_r):
    return v0 + (gamma_dot / eta_r) * x

  # Select empty bins
  sel = x[:,2] > 0
  
  # Nonlinear fit  
  try:
    popt, pcov = scop.curve_fit(profile, x[sel,0], x[sel,1], sigma=x[sel,2], p0=[0.1, 1])
    return popt[1], np.sqrt(pcov[1,1])
  except: 
    return 1, 1e+25


def compute_velocity_slope(x):
  '''
  Compute visocity from velocity profile.
  '''
  def profile(x, v0, slope):
    return v0 + slope * x

  # Select empty bins
  sel = x[:,2] > 0
  
  # Nonlinear fit  
  try:
    popt, pcov = scop.curve_fit(profile, x[sel,0], x[sel,1], sigma=x[sel,2], p0=[0, 0])
    return popt[1], np.sqrt(pcov[1,1])
  except: 
    return 0, 1e+25


def nonlinear_fit(x, y, func, sigma=None, p0=None, save_plot_name=None):
  '''
  Nonlinear fit.
  '''

  # Do non-linear fit
  popt, pcov = scop.curve_fit(func, x, y, sigma=sigma, p0=p0)

  # Compute adjusted R**2
  degrees_freedom = x.size - 1
  degrees_freedom_fit = x.size - 1 - popt.size
  residual = y - func(x, *popt)
  sum_squares_residual = np.sum(residual**2)
  y_mean = np.sum(y) / y.size
  sum_squares_total = np.sum((y - y_mean)**2)
  R2_adjusted = 1 - (sum_squares_residual / degrees_freedom_fit) / (sum_squares_total / degrees_freedom)

  # Plot data and func
  if save_plot_name is not None:
    # Create panel
    fig, axes = plt.subplots(1, 1, figsize=(5,5))

    # Plot
    plt.errorbar(x, y, sigma, marker='s', mfc='green', mec='green', ms=4, mew=0, label='data')

    # Set axes and legend
    plt.plot(x, func(x, *popt), 'r-', label='fit')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    fig.tight_layout()
  
    # Save to pdf and png
    plt.savefig(save_plot_name, format='png') 

  return popt, pcov, R2_adjusted
