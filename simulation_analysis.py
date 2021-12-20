import numpy as np
import scipy.stats as scst
import scipy.optimize as scop
import scipy.spatial as scsp
import sys
try:
  import matplotlib.pyplot as plt
except ImportError as e:
  print(e)
try:
  from numba import njit, prange
except ImportError as e:
  njit=None
  print(e)



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


def read_vertex(name):
  '''
  It reads a vertex file of the rigid bodies and return
  the coordinates as a numpy array with shape (Nblobs, 3).
  '''
  comment_symbols = ['#']   
  coor = []
  with open(name, 'r') as f:
    i = 0
    for line in f:
      # Strip comments
      if comment_symbols[0] in line:
        line, comment = line.split(comment_symbols[0], 1)

      # Ignore blank lines
      line = line.strip()
      if line != '':
        if i == 0:
          Nblobs = int(line.split()[0])
        else:
          location = np.fromstring(line, sep=' ')
          coor.append(location)
        i += 1

  coor = np.array(coor)
  return coor

    
def compute_velocities(x, dt=1, frame_rate=1):
  '''
  Compute velocities between frames.

  Return with shape = (num_frames, num_bodies, 6).
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


def compute_histogram(sample, column_sample=0, num_intervales=10, xmin=0, xmax=1, N_avg=1, header='', name=None):
  '''
  Compute histogram averages. 
  '''
  
  # Loop over steps
  # h = np.zeros((num_intervales, 2))

  # Created histogram
  h, h_edges = np.histogram(sample[:, column_sample], num_intervales, range=(xmin, xmax), density=True)
  hf, h_edges = np.histogram(sample[:, column_sample], num_intervales, range=(xmin, xmax), density=False)

  # Set histogram
  hist = np.zeros((h.size, 3))
  hist[:,0] = (h_edges[0:-1] + h_edges[1:]) / 2
  hist[:,1] = h
  hist[:,2] = hf

  # Save to a file
  if name is not None:
    np.savetxt(name, hist, header=header)

  return hist


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
  R2_adjusted = 1 - (sum_squares_residual / degrees_freedom_fit) / (np.maximum(sum_squares_total, 1e-12) / degrees_freedom)

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


def rotation_matrix(theta):
    ''' 
    Return the rotation matrix representing rotation by this quaternion.
    '''
    theta /= np.linalg.norm(theta)
    s = theta[0]
    p = theta[1:4]    
    diag = s**2 - 0.5
    return 2.0 * np.array([[p[0]**2+diag,     p[0]*p[1]-s*p[2], p[0]*p[2]+s*p[1]], 
                           [p[1]*p[0]+s*p[2], p[1]**2+diag,     p[1]*p[2]-s*p[0]],
                           [p[2]*p[0]-s*p[1], p[2]*p[1]+s*p[0], p[2]**2+diag]])


def save_xyz(x, r_vectors, name, num_frames=1, letter='O', body_frame_vector=None, body_vector=None, global_vector=None, header=''):
  '''
  Save xyz file.
  '''
  # Save M frames
  M = x.shape[0] if x.shape[0] < num_frames else num_frames
  Nblobs = x.shape[1] * r_vectors.size // 3
  file_output = open(name, 'w')

  # Loop over frames
  for i, xi in enumerate(x[0:M]):
    file_output.write(str(Nblobs) + '\n# ' + header + '\n')
        
    if body_vector is not None:
      vr = body_vector[i]
      
    if global_vector is not None:
      vg = global_vector[i]
    
    for j, y in enumerate(xi):
      theta = y[3:8]
      R = rotation_matrix(theta)
      r = np.dot(r_vectors, R.T) + y[0:3]
      r = r.reshape((r.size // 3, 3))

      if body_frame_vector is not None:
        v = np.dot(body_frame_vector, R.T)
        
      for k, ri in enumerate(r):     
        file_output.write(letter + ' %s %s %s ' % (ri[0], ri[1], ri[2]))

        if body_frame_vector is not None:
          np.savetxt(file_output, v[j], newline=' ')

        if body_vector is not None:
          np.savetxt(file_output, vr[j], newline=' ')

        if global_vector is not None:
          np.savetxt(file_output, vg, newline=' ')
        file_output.write('\n')
                  
  return 


def save_dat(x, t, i, name, header=''):
  '''
  Save body i as a data file.
  '''
  result = np.zeros((t.size, 8))
  result[:,0] = t
  result[:,1:] = x[:,i]
  np.savetxt(name, result, header=header)
  return

  
def project_to_periodic_image(r, L):
    '''
    Project a vector r to the minimal image representation
    of size L=(Lx, Ly, Lz) and with a corner at (0,0,0). If 
    any dimension of L is equal or smaller than zero the 
    box is assumed to be infinite in that direction.
    
    If one dimension is not periodic shift all coordinates by min(r[:,i]) value.
    '''
    if L is not None:
      for i in range(3):
        if(L[i] > 0):
          r[:,i] = r[:,i] - (r[:,i] // L[i]) * L[i]
        else:
          ri_min =  np.min(r[:,i])
          if ri_min < 0:
            r[:,i] -= ri_min
    return r


@njit(parallel=False, fastmath=True)
def gr_numba(r_vectors, L, list_of_neighbors, offsets, rcut, nbins, dim, Nblobs_body):
  '''
  This function compute the gr for one snapshot.
  '''
  N = r_vectors.size // 3
  r_vectors = r_vectors.reshape((N, 3))
  dbin = rcut / nbins
  gr = np.zeros((nbins, 2))

  # Copy arrays
  rx_vec = np.copy(r_vectors[:,0])
  ry_vec = np.copy(r_vectors[:,1])
  rz_vec = np.copy(r_vectors[:,2])
  Lx = L[0]
  Ly = L[1]
  Lz = L[2]

  for i in prange(N):
    i_body = i // Nblobs_body
    for k in range(offsets[i+1] - offsets[i]):
      j = list_of_neighbors[offsets[i] + k]
      j_body = j // Nblobs_body
      if (i >= j) or (i_body == j_body):
        continue
      rx = rx_vec[j] - rx_vec[i]
      ry = ry_vec[j] - ry_vec[i]
      rz = rz_vec[j] - rz_vec[i]

      # Use distance with PBC
      if Lx > 0:
        rx = rx - int(rx / Lx + 0.5 * (int(rx>0) - int(rx<0))) * Lx
      if Ly > 0:
        ry = ry - int(ry / Ly + 0.5 * (int(ry>0) - int(ry<0))) * Ly
      if Lz > 0:
        rz = rz - int(rz / Lz + 0.5 * (int(rz>0) - int(rz<0))) * Lz

      # Compute distance
      if dim == '2d' :
        r_norm = np.sqrt(rx*rx + ry*ry)
      else:
        r_norm = np.sqrt(rx*rx + ry*ry + rz*rz)
      xbin = int(r_norm / dbin)
      if xbin < nbins:
        gr[xbin, :] += 2

  return gr


def radial_distribution_function(x, num_frames, rcut=1.0, nbins=100, r_vectors=None, L=np.ones(3), dim='3d', name=None, header=''):
  '''
  Compute radial distribution function between bodies or blobs.
  It assumes all bodies are the same.

  dim=3d, 2d or q2d (treat as 3d but normalize as 2d).
  '''
  # Prepare variables
  M = x.shape[0] if x.shape[0] < num_frames else num_frames
  if r_vectors is not None:
    Nblobs_body = r_vectors.size // 3
    N = x.shape[1] * Nblobs_body
  else:
    Nblobs_body = 0
    N = x.shape[1] 
  dbin = rcut / nbins
  gr = np.zeros((nbins, 3))
  gr[:,0] = np.linspace(0, rcut, num=nbins+1)[:-1] + dbin / 2

  # Loop over frames
  for i, xi in enumerate(x[0:M]):
    if r_vectors is None:
      z = xi[:,0:3]
    else:
      z = np.zeros((N, 3))
      for j, y in enumerate(xi):
        theta = y[3:8]
        R = rotation_matrix(theta)
        r = np.dot(r_vectors, R.T) + y[0:3]
        r = r.reshape((r.size // 3, 3))
        z[Nblobs_body * j: Nblobs_body * (j+1)] = r
    
    # Project to PBC
    z = project_to_periodic_image(np.copy(z), L)

    # Set box dimensions for PBC
    if L[0] > 0 or L[1] > 0 or L[2] > 0:
      boxsize = np.zeros(3)
      for i in range(3):
        if L[i] > 0:
          boxsize[i] = L[i]
        else:
          boxsize[i] = (np.max(r_vectors[:,i]) - np.min(r_vectors[:,i])) + d_max * 10
    else:
      boxsize = None   

    # Build tree 
    tree = scsp.cKDTree(z, boxsize=boxsize)
    pairs = tree.query_ball_tree(tree, rcut)
    offsets = np.zeros(len(pairs)+1, dtype=int)
    for j in range(len(pairs)):
      offsets[j+1] = offsets[j] + len(pairs[j])
    list_of_neighbors = np.concatenate(pairs).ravel()

    gri = gr_numba(z, L, list_of_neighbors, offsets, rcut, nbins, dim, Nblobs_body)
    gr[:,1:3] += gri[:]
      
  # Normalize gr
  if dim == '3d':
    factor = (4 * np.pi / 3) * ((gr[:,0] + dbin)**3 - (gr[:,0] - dbin)**3) * M * N**2 / (L[0] * L[1] * L[2])
    gr[:,1] = gr[:,1] / factor
  else:
    factor = np.pi * ((gr[:,0] + dbin)**2 - (gr[:,0] - dbin)**2) * M * N**2 / (L[0] * L[1])
    gr[:,1] = gr[:,1] / factor

  # Save gr
  if name is not None:
    if len(header) == 0:
      header='Columns: r, gr density, gr count number'
      np.savetxt(name, gr, header=header)

  return gr
