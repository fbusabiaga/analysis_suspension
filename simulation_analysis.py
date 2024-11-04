import numpy as np
import scipy.stats as scst
import scipy.optimize as scop
import scipy.spatial as scsp
import sys
import time

try:
  import matplotlib.pyplot as plt
except ImportError as e:
  print(e)

try:
  from numba import njit, prange
except ImportError as e:
  njit=None
  print(e)

try:
  import sklearn.cluster as skcl
except ImportError as e:
  print(e)

# Try to import the visit_writer (boost implementation)
try:
  import visit_writer as visit_writer
except ImportError as e:
  print(e)
  pass


# Static Variable decorator for calculating acceptance rate.
def static_var(varname, value):
  def decorate(func):
    setattr(func, varname, value)
    return func
  return decorate


@static_var('timers', {})   
def timer(name, print_one = False, print_all = False, clean_all = False, file_name = None):
  '''
  Timer to profile the code. It measures the time elapsed between successive
  calls and it prints the total time elapsed after sucesive calls.  
  '''
  if name is not None:
    if name not in timer.timers:
      timer.timers[name] = (0, time.time())
    elif timer.timers[name][1] is None:
      time_tuple = (timer.timers[name][0],  time.time())
      timer.timers[name] = time_tuple
    else:
      time_tuple = (timer.timers[name][0] + (time.time() - timer.timers[name][1]), None)
      timer.timers[name] = time_tuple
      if print_one is True:
        print(name, ' = ', timer.timers[name][0])

  if print_all is True and len(timer.timers) > 0:
    print('\n')
    col_width = max(len(key) for key in timer.timers)
    for key in sorted(timer.timers):
      print("".join(key.ljust(col_width)), ' = ', timer.timers[key][0])
    if file_name is not None:
      with open(file_name, 'w') as f_handle:
        for key in sorted(timer.timers):
          f_handle.write("".join(key.ljust(col_width)) + ' = ' + str(timer.timers[key][0]) + '\n')
      
  if clean_all:
    timer.timers = {}
  return


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
        if num_lines == 1:
          number_variables = len(line.split())
        num_lines += 1
  except OSError:
    return np.array([])

  # Set array 
  num_frames = num_lines // (N + 1) 
  x = np.zeros((num_frames, N, number_variables)) 

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


def read_config_list(names, print_name=False):
  '''
  Read list of config files x_0, x_1, x_2 ... 
  It is assumed that the first snapshot of the file x_{n+1} is the same as the last
  snapshot of the file x_n.

  The config is stored in an array of shape (num_frames, num_bodies, 7).
  '''

  # Loop over files
  x = []
  for j, name in enumerate(names):
    if print_name:
      print('name_config = ', name)

    xj = read_config(name)
    if j == 0 and xj.size > 0:
      x.append(xj)
    elif xj.size > 0:
      x.append(xj[1:])

  # Concatenate configs
  x = np.concatenate([xi for xi in x])
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
  return coor[:,0:3]


def read_vertex_file_list(name_files, path=None):
  '''
  It reads a file with a list of vertex files.
  It returns a list, each element the coordinates of the blobs in a rigid body.
  '''
  comment_symbols = ['#']   
  struct_ref_config = []
  with open(name_files) as f:
    for line in f:
      # Strip comments
      if comment_symbols[0] in line:
        line, comment = line.split(comment_symbols[0], 1)

      # Ignore blank lines
      line = line.strip()
      if line != '':
        if path is None:
          name = line.split()[0]
        else:
          name = path + line.split()[0]
        struct = read_vertex(name)
        struct_ref_config.append(struct)      

  return struct_ref_config


def read_stress(name):
  '''
  Read stress and store in an array of shape (num_frames, num_bodies, 9).
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
  x = np.zeros((num_frames, N, 9))

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


def compute_linear_velocities(x, dt=1, frame_rate=1):
  '''
  Compute velocities between frames.

  Return with shape = (num_frames, num_bodies, 3).
  '''
  num_frames = x.shape[0]
  N = x.shape[1]
  dt_frames = frame_rate * dt
  v = np.zeros((num_frames-frame_rate, N, 3))

  # Loop over frames and bodies
  for i in range(num_frames - frame_rate):
    # Linear velocity
    v[i,:,0:3] = (x[i+frame_rate, :, 0:3] - x[i, :, 0:3]) / dt_frames

  # Return velocities
  return v


def compute_histogram_from_frames(sample, value=None, column_sample=0, column_value=0, num_intervales=10, xmin=0, xmax=1, N_avg=1, file_prefix=None):
  '''
  Compute histogram averages. Use column_sample to select the bin and column_value to compute the average.
  If value is None use a density histogram.
  '''
  
  # Loop over steps
  if value is None:
    N_frames = sample.shape[0]
  else:
    N_frames = np.minimum(sample.shape[0], value.shape[0])
  N_steps = N_frames // N_avg
  h = np.zeros((N_steps, num_intervales, 3))
  for i in range(N_steps):
    mean_avg = np.zeros(num_intervales)
    std_avg = np.zeros(num_intervales)
    counter = np.zeros(num_intervales)

    for j in range(N_avg):
      # Created binned average
      if value is None:
        mean, bin_edges = np.histogram(sample[i*N_avg + j, :, column_sample], num_intervales, range=(xmin, xmax), density=True)
        bin_edges = bin_edges.reshape(1, bin_edges.size)
      else:
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


def compute_histogram(sample, column_sample=0, num_intervales=10, xmin=0, xmax=1, scale='linear', max_points_bin=10, max_levels=10, min_width=1, header='', name=None):
  '''
  Compute histogram averages.  
  '''
  # Reshape if necessary 
  if sample.ndim == 1:
    sample = sample.reshape(sample.size, 1)  

  if scale == 'adaptive':
    # Set preliminary edges with enough points within bin
    if not isinstance(num_intervales, np.ndarray):
      num_intervales = xmin + np.arange(num_intervales + 1) * (xmax - xmin) / num_intervales
    counter = 0
    while True:
      hf, h_edges = np.histogram(sample[:, column_sample], num_intervales, range=(xmin, xmax), density=False)
      sel = np.zeros(h_edges.size, dtype=bool)
      sel[0:-1] = hf > max_points_bin
      if np.all(~sel) or counter >= max_levels:
        break
      h_new_edges = (h_edges[sel] + h_edges[np.where(sel==True)[0]+1]) / 2
      num_intervales = np.insert(num_intervales, np.where(sel==True)[0]+1, h_new_edges)
      counter += 1

    # If edges are too close remove some
    position = 0
    while True:
      if position < num_intervales.size - 2:
        if num_intervales[position+1] - num_intervales[position] < min_width:
          num_intervales = np.delete(num_intervales, position+1)
        else:
          position += 1
      else:
        break
    num_intervales = np.unique(np.around(num_intervales, decimals=2))

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


def rotation_matrix_from_vector_and_angle(u, theta):
  ''' 
  Return the rotation matrix representing rotation by a unit vector.
  '''
  u /= np.linalg.norm(u)
  c = np.cos(theta)
  s = np.sin(theta)
  return np.array([[u[0]**2 * (1 - c) + c,            u[0] * u[1] * (1 - c) - u[2] * s, u[0] * u[2] * (1 - c) + u[1] * s],
                   [u[0] * u[1] * (1 - c) + u[2] * s, u[1]**2 * (1 - c) + c,            u[1] * u[2] * (1 - c) - u[0] * s],
                   [u[0] * u[2] * (1 - c) - u[1] * s, u[1] * u[2] * (1 - c) + u[0] * s, u[2]**2 * (1 - c) + c]])

  
def save_xyz(x, r_vectors, name, num_frames=1, letter='O', articulated=False, body_frame_vector=None, body_vector=None, global_vector=None,
             blob_vector_constant=None, body_frame_blob_vector=None, periodic_length=np.zeros(3), frame_body=-1, header=''):
  '''
  Save xyz file.
  '''
  # Save M frames
  M = x.shape[0] if x.shape[0] < num_frames else num_frames

  # Check if it is an articulated body
  if articulated:
    Nrigid = len(r_vectors)
    Nblobs = x.shape[1] * np.sum([r.size // 3 for r in r_vectors]) // Nrigid
  else:
    Nblobs = x.shape[1] * r_vectors.size // 3
    Nrigid = 1
    r_vectors = r_vectors.reshape((1, r_vectors.size // 3, 3))

  # Open output file
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
      q = y[0:3]
      R = rotation_matrix(theta)
      if frame_body > -1:
        q = np.dot(rotation_matrix(xi[frame_body, 3:8]).T, (y[0:3] - xi[frame_body, 0:3]))
        R = np.dot(rotation_matrix(xi[frame_body, 3:8]).T, R)
      r = np.dot(r_vectors[j % Nrigid], R.T) + q
      r = r.reshape((r.size // 3, 3))

      # Project to PBC 
      if np.any(periodic_length > 0):
        r = project_to_periodic_image(r, periodic_length, shift=False)

      if body_frame_vector is not None:
        v = np.dot(body_frame_vector, R.T)

      if body_frame_blob_vector is not None:
        bfbv = np.dot(body_frame_blob_vector[j], R.T)
        
      for k, ri in enumerate(r):
        letter = str(j % 26)
        file_output.write(letter + ' %s %s %s ' % (ri[0], ri[1], ri[2]))

        if body_frame_vector is not None:
          np.savetxt(file_output, v[j], newline=' ')

        if body_frame_blob_vector is not None:
          np.savetxt(file_output, bfbv[k], newline=' ')

        if body_vector is not None:
          np.savetxt(file_output, vr[j], newline=' ')

        if global_vector is not None:
          np.savetxt(file_output, vg, newline=' ')

        if blob_vector_constant is not None:
          to_save = blob_vector_constant[j][k]
          np.savetxt(file_output, to_save.reshape((1, to_save.size)), newline=' ')
          
        file_output.write('\n')
  file_output.flush()
                  
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

  
def project_to_periodic_image(r, L, shift=True):
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
        elif shift:
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
    Nblobs_body = 1
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
      for j in range(3):
        if L[j] > 0:
          boxsize[j] = L[j]
        else:
          boxsize[j] = (np.max(z[:,j]) - np.min(z[:,j])) + rcut * 10
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
    factor = (4 * np.pi / 3) * ((gr[:,0] + dbin / 2)**3 - (gr[:,0] - dbin / 2)**3) * M * N**2 / (L[0] * L[1] * L[2])
    gr[:,1] = gr[:,1] / factor
  else:
    factor = np.pi * ((gr[:,0] + dbin / 2)**2 - (gr[:,0] - dbin / 2)**2) * M * N**2 / (L[0] * L[1])
    gr[:,1] = gr[:,1] / factor

  # Save gr
  if name is not None:
    if len(header) == 0:
      header='Columns: r, gr density, gr count number'
      np.savetxt(name, gr, header=header)

  return gr


@njit(parallel=False, fastmath=True)
def pair_distribution_numba(r_vectors, L, list_of_neighbors, offsets, rcutx, rcuty, rcutz, nbinsx, nbinsy, nbinsz, dbinx, dbiny, dbinz, dim, Nblobs_body, Lx_wall, Ly_wall, Lz_wall, offset_walls):
  '''
  This function compute the pair distribution function g(x,y,z) for one snapshot.
  '''
  N = r_vectors.size // 3
  r_vectors = r_vectors.reshape((N, 3))
  gr = np.zeros((nbinsz, nbinsy, nbinsx))

  # Copy arrays
  rx_vec = np.copy(r_vectors[:,0])
  ry_vec = np.copy(r_vectors[:,1])
  rz_vec = np.copy(r_vectors[:,2])
  Lx = L[0]
  Ly = L[1]
  Lz = L[2]

  # Set offset distance from walls
  if offset_walls:
    LxW = Lx_wall + np.array([rcutx, -rcutx])
    LyW = Ly_wall + np.array([rcuty, -rcuty])
    LzW = Lz_wall + np.array([rcutz, -rcutz])
  else:
    LxW = Lx_wall 
    LyW = Ly_wall 
    LzW = Lz_wall

  for i in prange(N):
    i_body = i // Nblobs_body

    # Skipt if close to a wall
    if rx_vec[i] < LxW[0] or rx_vec[i] > LxW[1] or ry_vec[i] < LyW[0] or ry_vec[i] > LyW[1] or rz_vec[i] < LzW[0] or rz_vec[i] > LzW[1]:
      continue
    
    for k in range(offsets[i+1] - offsets[i]):
      j = list_of_neighbors[offsets[i] + k]
      j_body = j // Nblobs_body
      if (i == j) or (i_body == j_body):
        continue
      rx = rx_vec[j] - rx_vec[i]
      ry = ry_vec[j] - ry_vec[i]
      rz = rz_vec[j] - rz_vec[i]

      # Use distance with PBC
      if Lx >= 0:
        rx = rx - int(rx / Lx + 0.5 * (int(rx>0) - int(rx<0))) * Lx if Lx > 0 else 0
      if Ly >= 0:
        ry = ry - int(ry / Ly + 0.5 * (int(ry>0) - int(ry<0))) * Ly if Ly > 0 else 0
      if Lz >= 0:
        rz = rz - int(rz / Lz + 0.5 * (int(rz>0) - int(rz<0))) * Lz if Lz > 0 else 0
        
      # Compute bins
      if rcutx > 0:
        xbin = int((rx + rcutx) / dbinx) if rx >= -rcutx else -1
      else:
        xbin = 0
      if rcuty > 0:
        ybin = int((ry + rcuty) / dbiny) if ry >= -rcuty else -1
      else:
        ybin = 0
      if rcutz > 0:        
        zbin = int((rz + rcutz) / dbinz) if rz >= -rcutz else -1
      else:
        zbin = 0
      dbin = int(np.sqrt(rx**2 + ry**2) / dbinx)

      # Update gr
      if dim == '3d':
        if (xbin >= 0) and (xbin < nbinsx) and (ybin >= 0) and (ybin < nbinsy) and (zbin >= 0) and (zbin < nbinsz):
          # Beware x is the fast axis in visit
          gr[zbin, ybin, xbin] += 1.0
      elif dim == '2d_radial':
        if (dbin < nbinsx) and (zbin >= 0) and (zbin < nbinsz):
          # Beware x is the fast axis in visit
          gr[zbin, 0, dbin] += 1.0
        
  return gr


def pair_distribution_function(x,
                               num_frames,
                               rcut=1.0,
                               nbinsx=10,
                               nbinsy=10,
                               nbinsz=10,
                               r_vectors=None,
                               L=np.ones(3),
                               offset_walls=False,
                               dim='3d',
                               name=None,
                               Lx_wall=np.array([-np.inf, np.inf]),
                               Ly_wall=np.array([-np.inf, np.inf]),
                               Lz_wall=np.array([-np.inf, np.inf])):
  '''
  Compute pair distribution function, g(x,y,z), between bodies or blobs.
  It assumes all bodies are the same.

  Some inputs:
  r_vectors = if None use body positions otherwise use blobs.
  L = periodic dimensions. Use 0 if the system is infinite or have walls along one axis.
  offset_walls = if True do not use particles near than rcut from walls.
  Lx_wall = position of the walls along x axis. Use +/- np.inf if there are not walls.
  dim=3d or 2d. 
  '''
  # Prepare variables
  M = x.shape[0] if x.shape[0] < num_frames else num_frames
  if r_vectors is not None:
    Nblobs_body = r_vectors.size // 3
    N = x.shape[1] * Nblobs_body
  else:
    Nblobs_body = 1
    N = x.shape[1]
    
  # Note that the bins should go from negative r to positive r so the factor 2 in dbin
  if dim == '3d':
    gr = np.zeros((nbinsz, nbinsy, nbinsx))
    rcutx = rcut if rcut < L[0] / 2 else L[0] / 2
    rcuty = rcut if rcut < L[1] / 2 else L[1] / 2
    rcutz = rcut if rcut < L[2] / 2 else L[2] / 2
    dbinx = 2 * rcutx / nbinsx if nbinsx > 0 else 0
    dbiny = 2 * rcuty / nbinsy if nbinsy > 0 else 0
    dbinz = 2 * rcutz / nbinsz if nbinsz > 0 else 0
    rcut_tree = np.sqrt(rcutx**2 + rcuty**2 + rcutz**2)

  elif dim == '2d':
    gr = np.zeros((nbins, nbins))
    rcut_tree = rcut * np.sqrt(2.0)
  elif dim == '2d_radial':
    gr = np.zeros((nbinsz, 1, nbinsx))
    rcutx = rcut if rcut < L[0] / 2 else L[0] / 2
    rcuty = rcut if rcut < L[1] / 2 else L[1] / 2
    rcutz = rcut if rcut < L[2] / 2 else L[2] / 2
    dbinx = 1 * rcutx / nbinsx if nbinsx > 0 else 0
    dbiny = 1
    dbinz = 2 * rcutz / nbinsz if nbinsz > 0 else 0
    rcut_tree = np.sqrt(rcutx**2 + rcuty**2 + rcutz**2)
  else:
    print('Wrong dimensions, dim = ', dim,' is not valid')
    return None

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
      for j in range(3):
        if L[j] > 0:
          boxsize[j] = L[j]
        else:
          boxsize[j] = (np.max(z[:,j]) - np.min(z[:,j])) + rcut_tree * 10
    else:
      boxsize = None

    # Build tree 
    tree = scsp.cKDTree(z, boxsize=boxsize)
    pairs = tree.query_ball_tree(tree, rcut_tree)
    offsets = np.zeros(len(pairs)+1, dtype=int)
    for j in range(len(pairs)):
      offsets[j+1] = offsets[j] + len(pairs[j])
    list_of_neighbors = np.concatenate(pairs).ravel()

    gri = pair_distribution_numba(z,
                                  L,
                                  list_of_neighbors,
                                  offsets,
                                  rcutx,
                                  rcuty,
                                  rcutz,
                                  nbinsx,
                                  nbinsy,
                                  nbinsz,
                                  dbinx,
                                  dbiny,
                                  dbinz,
                                  dim,
                                  Nblobs_body,
                                  Lx_wall,
                                  Ly_wall,
                                  Lz_wall,
                                  offset_walls=offset_walls)
    gr += gri
    
  # Normalize gr 
  if dim == '3d':
    Lx = L[0] if np.any(np.isinf(Lx_wall)) else Lx_wall[1] - Lx_wall[0]
    Ly = L[1] if np.any(np.isinf(Ly_wall)) else Ly_wall[1] - Ly_wall[0]
    Lz = L[2] if np.any(np.isinf(Lz_wall)) else Lz_wall[1] - Lz_wall[0]
    factor = M*N*(N-1) * (dbinx if dbinx > 0 else 1) * (dbiny if dbiny > 0 else 1) * (dbinz if dbinz > 0 else 1) / ((Lx if Lx > 0 else 1) * (Ly if Ly > 0 else 1) * (Lz if Lz > 0 else 1))
    gr = gr / factor
  elif dim == '2d_radial':
    Lx = L[0] if np.any(np.isinf(Lx_wall)) else Lx_wall[1] - Lx_wall[0]
    Ly = L[1] if np.any(np.isinf(Ly_wall)) else Ly_wall[1] - Ly_wall[0]
    Lz = L[2] if np.any(np.isinf(Lz_wall)) else Lz_wall[1] - Lz_wall[0]
    factor = M * N * (N-1) * np.pi * (np.arange(1, nbinsx + 1)**2 - np.arange(0, nbinsx)**2) * dbinx**2 * dbinz / (Lx * Ly * Lz)
    gr = gr / factor[None,None,:]
  else:
    Lx = L[0] if np.any(np.isinf(Lx_wall)) else Lx_wall[1] - Lx_wall[0]
    Ly = L[1] if np.any(np.isinf(Ly_wall)) else Ly_wall[1] - Ly_wall[0]
    factor = M * N * (N-1) * dbin**2 / (Lx * Ly)
    gr = gr[:,1] / factor

  # Save gr
  if name is not None:
    # Prepara data for VTK writer 
    variables = [np.reshape(gr, gr.size)] 
    dims = np.array([nbinsx+1, nbinsy+1, nbinsz+1], dtype=np.int32) 
    nvars = 1
    vardims = np.array([1])
    centering = np.array([0])
    varnames = ['pair_distribution\0']
    if dim == '2d_radial':
      grid_x = np.arange(nbinsx + 1) * dbinx
    else:
      grid_x = np.arange(nbinsx + 1) * dbinx - (nbinsx * 0.5) * dbinx
    grid_y = np.arange(nbinsy + 1) * dbiny - (nbinsy * 0.5) * dbiny
    grid_z = np.arange(nbinsz + 1) * dbinz - (nbinsz * 0.5) * dbinz

    # Write velocity field
    visit_writer.boost_write_rectilinear_mesh(name,      # File's name
                                              0,         # 0=ASCII,  1=Binary
                                              dims,      # {mx, my, mz}
                                              grid_x,     # xmesh
                                              grid_y,     # ymesh
                                              grid_z,     # zmesh
                                              nvars,     # Number of variables
                                              vardims,   # Size of each variable, 1=scalar, velocity=3*scalars
                                              centering, # Write to cell centers of corners
                                              varnames,  # Variables' names
                                              variables) # Variables
    
  return gr


def correlation(g,h):
  '''
  Compute the correlation between two 1D functions using fft.
  
  Corr(g,h) = sum(g(t+tau) * h(tau)) = IFFT( FFT(g) * (FFT(h))^*) / Normalization
  
  Note that we have to padd the input array with zeros   because it is not necessarily periodic. 
  See Numerical Recipies in C.
  '''

  # Padd input array with zeros and compute FFT
  g_fft = np.fft.fft(np.concatenate([g, np.zeros(g.size)]))
  h_fft_conj = np.conj(np.fft.fft(np.concatenate([h, np.zeros(h.size)])))

  # Compute product, IFFT and normalize
  return np.fft.ifft(g_fft * h_fft_conj)[0:g.size] / np.arange(len(g), 0, -1) 


def rotational_correlation(x, dt, axis0=None, Corr_steps=None, output_name=None, header=''):
  '''
  Compute the rotational MSD from the trajectory
  '''
  # Init variables
  num_bodies = x.shape[1]
  ex = np.array([1, 0, 0])
  ey = np.array([0, 1, 0])
  ez = np.array([0, 0, 1])

  # Allocate MSD memory
  if Corr_steps is None:
    Corr_steps = x.shape[0]
  else:
    Corr_steps = x.shape[0] if x.shape[0] < Corr_steps else Corr_steps
  Corr_average = np.zeros((Corr_steps))
  Corr_std = np.zeros((Corr_steps))

  # Compute correlations
  counter = 0
  for body in range(num_bodies):
    # Allocate memory
    Corr = np.zeros((Corr_steps))
    axis1 = np.zeros((x.shape[0], 3))
    axis2 = np.zeros((x.shape[0], 3))
    axis3 = np.zeros((x.shape[0], 3))

    # Loop over time to compute body vectors in the laboratory frame of ref.
    for ti in range(x.shape[0]):
      # 1. Get rotation matrix
      theta = x[ti,body,3:8]
      R = rotation_matrix(theta)

      # 2. Get body axes
      axis1[ti] = np.dot(R, ex) if axis0 is None else np.dot(R, axis0)
      axis2[ti] = np.dot(R, ey)
      axis3[ti] = np.dot(R, ez)

    # Compute correlation
    if axis0 is not None:
      corr1_x = np.real(correlation(axis1[:,0], axis1[:,0]))
      corr1_y = np.real(correlation(axis1[:,1], axis1[:,1]))
      corr1_z = np.real(correlation(axis1[:,2], axis1[:,2]))
      Corr = corr1_x + corr1_y + corr1_z
    else:
      corr1_x = np.real(correlation(axis1[:,0], axis1[:,0]))
      corr1_y = np.real(correlation(axis1[:,1], axis1[:,1]))
      corr1_z = np.real(correlation(axis1[:,2], axis1[:,2]))
      corr2_x = np.real(correlation(axis2[:,0], axis2[:,0]))
      corr2_y = np.real(correlation(axis2[:,1], axis2[:,1]))
      corr2_z = np.real(correlation(axis2[:,2], axis2[:,2]))
      corr3_x = np.real(correlation(axis3[:,0], axis3[:,0]))
      corr3_y = np.real(correlation(axis3[:,1], axis3[:,1]))
      corr3_z = np.real(correlation(axis3[:,2], axis3[:,2]))   
      Corr = (corr1_x + corr1_y + corr1_z + corr2_x + corr2_y + corr2_z + corr3_x + corr3_y + corr3_z) / 3
          
    # Compute average Corr
    if np.min(Corr) < 0.5:
      Corr_average += (Corr[0:Corr_steps] - Corr_average) / (counter + 1)
      Corr_std += counter * (Corr[0:Corr_steps] - Corr_average)**2 / (counter + 1)
      counter += 1

  # Normalize std 
  Corr_std = np.sqrt(Corr_std / np.maximum(1, counter - 1))
  print('counter = ', counter)
  
  # Save to file
  if output_name is not None:
    if len(header) == 0:
      header = 'Columns: linear Corr (1 terms), Corr std (1 terms)'
      result = np.zeros((Corr_steps, 3))
      result[:,0] = np.arange(Corr_steps) * dt
      result[:,1] = Corr_average[0:Corr_steps]
      result[:,2] = Corr_std[0:Corr_steps]      
      np.savetxt(output_name, result, header=header)
  
  return Corr_average, Corr_std


def msd_rotational(x, dt, MSD_steps=None, step_size=1, output_name=None, header=''):
  '''
  Compute the rotational MSD from the trajectory
  '''
  # Init variables
  num_bodies = x.shape[1]
  ex = np.array([1, 0, 0])
  ey = np.array([0, 1, 0])
  ez = np.array([0, 0, 1])

  # Allocate MSD memory
  if MSD_steps is None:
    MSD_steps = x.shape[0]
  else:
    MSD_steps = x.shape[0] if x.shape[0] < MSD_steps else MSD_steps
  MSD_average = np.zeros((MSD_steps // step_size, 3, 3))
  MSD_std = np.zeros((MSD_steps // step_size, 3, 3))

  # Compute correlations
  for body in range(num_bodies):
    # Allocate memory
    MSD = np.zeros((MSD_steps // step_size, 3, 3))
    delta_u = np.zeros((x.shape[0], 3))
    
    # 2. Get body axes at time zero
    u1 = np.zeros((x.shape[0], 3))
    u2 = np.zeros((x.shape[0], 3))
    u3 = np.zeros((x.shape[0], 3))    

    # Loop over time to compute body vectors in the laboratory frame of ref.
    for ti in range(x.shape[0]):
      # 1. Get rotation matrix
      theta = x[ti,body,3:8]
      R = rotation_matrix(theta)

      # 2. Get body axes
      u1[ti] = np.dot(R, ex)
      u2[ti] = np.dot(R, ey)
      u3[ti] = np.dot(R, ez)

    # Double loop to build MSD for one body
    taus = np.arange(MSD_steps // step_size) * step_size
    for tau in taus:
      for ti in range(x.shape[0] - tau):
        # Compute rotational displacment (see Kraft et al. 2013)
        u_hat = 0.5 * (np.cross(u1[ti], u1[ti + tau]) + np.cross(u2[ti], u2[ti + tau]) + np.cross(u3[ti], u3[ti + tau]))
        
        # Compute 6x6 MSD matrix
        MSD[tau // step_size] += (np.outer(u_hat, u_hat) - MSD[tau // step_size]) / (ti + 1)

    # Compute average MSD
    MSD_average += (MSD - MSD_average) / (body + 1)
    MSD_std += body * (MSD - MSD_average)**2 / (body + 1)
   
  # Normalize std 
  MSD_std = np.sqrt(MSD_std / np.maximum(1, body - 1))

  # Save to file
  if output_name is not None:
    if len(header) == 0:
      header = 'Columns: time, linear MSD (9 terms), MSD std (9 terms)'
      MSD_average = MSD_average.reshape(MSD.size // 9, 9)
      MSD_std = MSD_std.reshape(MSD.size // 9, 9)
      result = np.zeros((MSD_steps // step_size, 19))
      result[:,0] = np.arange(MSD_steps // step_size) * step_size * dt 
      result[:,1:10] = MSD_average
      result[:,10:19] = MSD_std
      np.savetxt(output_name, result, header=header)
  
  return MSD_average, MSD_std
  

def msd(x, dt, MSD_steps=None, output_name=None, header=''):
  '''
  Compute the translational MSD from the trajectory using FFT.

  For translational variables we use:
  N * MSD(tau) = sum((x(t+tau)-x(t))*(y(t+tau)-y(t))) = sum(x(t+tau)*y(t+tau)) + sum(x(t)*y(t)) - sum(x(t+tau)*y(t) - sum(x(t)*y(t+tau))

  and we can use FFT to compute the last two terms (cross-correlation).

  This code does not compute the rotational MSD.
  '''
  # Init variables
  num_bodies = x.shape[1]
 
  # Allocate MSD memory
  if MSD_steps is None:
    MSD_steps = x.shape[0]
  else:
    MSD_steps = x.shape[0] if x.shape[0] < MSD_steps else MSD_steps
  MSD = np.zeros((MSD_steps, 3, 3))
  MSD_average = np.zeros((MSD_steps, 3, 3))
  MSD_std = np.zeros((MSD_steps, 3, 3))

  # Compute correlations
  for body in range(num_bodies):
    corr_xx = np.real(correlation(x[:,body,0],x[:,body,0]))
    corr_xy = np.real(correlation(x[:,body,0],x[:,body,1]))
    corr_xz = np.real(correlation(x[:,body,0],x[:,body,2]))
    corr_yx = np.real(correlation(x[:,body,1],x[:,body,0]))
    corr_yy = np.real(correlation(x[:,body,1],x[:,body,1]))
    corr_yz = np.real(correlation(x[:,body,1],x[:,body,2]))
    corr_zx = np.real(correlation(x[:,body,2],x[:,body,0]))
    corr_zy = np.real(correlation(x[:,body,2],x[:,body,1]))
    corr_zz = np.real(correlation(x[:,body,2],x[:,body,2]))
       
    # Sum from t=0 to t=t_final-tau
    sum_xx = np.cumsum(x[:,body,0]*x[:,body,0])[::-1] / np.arange(len(corr_xx), 0, -1)
    sum_xy = np.cumsum(x[:,body,0]*x[:,body,1])[::-1] / np.arange(len(corr_xx), 0, -1)
    sum_xz = np.cumsum(x[:,body,0]*x[:,body,2])[::-1] / np.arange(len(corr_xx), 0, -1)
    sum_yx = np.cumsum(x[:,body,1]*x[:,body,0])[::-1] / np.arange(len(corr_xx), 0, -1)
    sum_yy = np.cumsum(x[:,body,1]*x[:,body,1])[::-1] / np.arange(len(corr_xx), 0, -1)
    sum_yz = np.cumsum(x[:,body,1]*x[:,body,2])[::-1] / np.arange(len(corr_xx), 0, -1)
    sum_zx = np.cumsum(x[:,body,2]*x[:,body,0])[::-1] / np.arange(len(corr_xx), 0, -1)
    sum_zy = np.cumsum(x[:,body,2]*x[:,body,1])[::-1] / np.arange(len(corr_xx), 0, -1)
    sum_zz = np.cumsum(x[:,body,2]*x[:,body,2])[::-1] / np.arange(len(corr_xx), 0, -1)

    # Sum from t=tau to t=t_final
    sum_xx_tau = np.cumsum(x[::-1,body,0]*x[::-1,body,0])[::-1] / np.arange(len(corr_xx), 0, -1)
    sum_xy_tau = np.cumsum(x[::-1,body,0]*x[::-1,body,1])[::-1] / np.arange(len(corr_xx), 0, -1)
    sum_xz_tau = np.cumsum(x[::-1,body,0]*x[::-1,body,2])[::-1] / np.arange(len(corr_xx), 0, -1)
    sum_yx_tau = np.cumsum(x[::-1,body,1]*x[::-1,body,0])[::-1] / np.arange(len(corr_xx), 0, -1)
    sum_yy_tau = np.cumsum(x[::-1,body,1]*x[::-1,body,1])[::-1] / np.arange(len(corr_xx), 0, -1)
    sum_yz_tau = np.cumsum(x[::-1,body,1]*x[::-1,body,2])[::-1] / np.arange(len(corr_xx), 0, -1)
    sum_zx_tau = np.cumsum(x[::-1,body,2]*x[::-1,body,0])[::-1] / np.arange(len(corr_xx), 0, -1)
    sum_zy_tau = np.cumsum(x[::-1,body,2]*x[::-1,body,1])[::-1] / np.arange(len(corr_xx), 0, -1)
    sum_zz_tau = np.cumsum(x[::-1,body,2]*x[::-1,body,2])[::-1] / np.arange(len(corr_xx), 0, -1)

    # Compute MSD
    MSD[:,0,0] = sum_xx_tau[:MSD_steps] + sum_xx[:MSD_steps] - 2.0 * corr_xx[:MSD_steps]
    MSD[:,0,1] = sum_xy_tau[:MSD_steps] + sum_xy[:MSD_steps] - corr_xy[:MSD_steps] - corr_yx[:MSD_steps]
    MSD[:,0,2] = sum_xz_tau[:MSD_steps] + sum_xz[:MSD_steps] - corr_xz[:MSD_steps] - corr_zx[:MSD_steps]
    MSD[:,1,0] = sum_yx_tau[:MSD_steps] + sum_yx[:MSD_steps] - corr_yx[:MSD_steps] - corr_xy[:MSD_steps]
    MSD[:,1,1] = sum_yy_tau[:MSD_steps] + sum_yy[:MSD_steps] - 2.0 * corr_yy[:MSD_steps]
    MSD[:,1,2] = sum_yz_tau[:MSD_steps] + sum_yz[:MSD_steps] - corr_yz[:MSD_steps] - corr_zy[:MSD_steps]
    MSD[:,2,0] = sum_zx_tau[:MSD_steps] + sum_zx[:MSD_steps] - corr_zx[:MSD_steps] - corr_xz[:MSD_steps]
    MSD[:,2,1] = sum_zy_tau[:MSD_steps] + sum_zy[:MSD_steps] - corr_zy[:MSD_steps] - corr_yz[:MSD_steps]
    MSD[:,2,2] = sum_zz_tau[:MSD_steps] + sum_zz[:MSD_steps] - 2.0 * corr_zz[:MSD_steps]
  
    # Compute MSD std
    MSD_std += body * (MSD - MSD_average) * (MSD - MSD_average) / float(body + 1)

    # Compute average MSD
    MSD_average += (MSD - MSD_average) / float(body + 1)    

  MSD_std = np.sqrt(MSD_std / np.maximum(1, num_bodies - 1))

  if output_name is not None:
    if len(header) == 0:
      header = 'Columns: linear MSD (9 terms)'
    MSD_average = MSD_average.reshape(MSD.size // 9, 9)
    MSD_std = MSD_std.reshape(MSD.size // 9, 9)
    result = np.zeros((MSD_steps, 10))
    result[:,0] = np.arange(MSD_steps) * dt
    result[:,1:10] = MSD_average[0:MSD_steps]
    np.savetxt(output_name, result, header=header)

  return MSD_average, MSD_std


@njit(parallel=False, fastmath=True)
def cluster_detection_numba(N, list_of_neighbors, offsets, Nblobs_body):
  '''
  Detect to which cluster belongs each body.
  '''
  cluster_i = np.ones(N // Nblobs_body, dtype=np.int32) * (N // Nblobs_body)

  # Loop over bodies
  for j in range(N):
    j_body = j // Nblobs_body
    min_body = np.minimum(j_body, cluster_i[j_body])
    neighbor = 0

    # Loop over neighbors: find lower body index
    for k in range(offsets[j+1] - offsets[j]):
      l = list_of_neighbors[offsets[j] + k]
      l_body = l // Nblobs_body
      if l_body == j_body:
        continue
      neighbor = 1
      if l_body < min_body or cluster_i[l_body] < min_body:
        min_body = np.minimum(l_body, cluster_i[l_body])
        
    # Set lower body index
    if neighbor > 0:
      cluster_i[j_body] = min_body
      for k in range(offsets[j+1] - offsets[j]):
        l = list_of_neighbors[offsets[j] + k]
        l_body = l // Nblobs_body
        cluster_i[l_body] = min_body
  return cluster_i


def cluster_detection(x, num_frames, rcut=1.0, r_vectors=None, L=np.ones(3), name=None, header=''):
  '''
  Detect clusters of bodies. Clusters are defined when blobs, or bodies if r_vectors=None, are nearer than rcut.

  It returns array of shape (num_frames, num_bodies) the value labels the cluster to which the body belongs. 
  We use the notation lable=-1 if the body does not belong to a cluster,
  and label=min(body_index_in_cluster).
  '''
  # Prepare variables
  M = x.shape[0] if x.shape[0] < num_frames else num_frames
  if r_vectors is not None:
    Nblobs_body = r_vectors.size // 3
    N = x.shape[1] * Nblobs_body
  else:
    Nblobs_body = 1
    N = x.shape[1]
  clusters = np.ones((M, x.shape[1]), dtype=int) * x.shape[1]

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
      for j in range(3):
        if L[j] > 0:
          boxsize[j] = L[j]
        else:
          boxsize[j] = (np.max(z[:,j]) - np.min(z[:,j])) + rcut * 10
    else:
      boxsize = None   

    # Build tree
    tree = scsp.cKDTree(z, boxsize=boxsize)
    pairs = tree.query_ball_tree(tree, rcut)
    offsets = np.zeros(len(pairs)+1, dtype=int)
    for j in range(len(pairs)):
      offsets[j+1] = offsets[j] + len(pairs[j])
    list_of_neighbors = np.concatenate(pairs).ravel()

    # Detect clusters in frame i
    cluster_i = cluster_detection_numba(N, list_of_neighbors, offsets, Nblobs_body)

    # Set values to -1 if body does not belong to any cluster
    sel = cluster_i == (N // Nblobs_body)
    cluster_i[sel] = -1
    clusters[i] = cluster_i
  
  return clusters


def map2d(x, y, variable, nx=10, ny=10, x_ends=None, y_ends=None, centered=False, n_avg=1, vmin=None, vmax=None, file_prefix=None):
  '''
  Compute color map of variable with (x,y) positions.
  '''
  # Compute limits for 2d plot
  if x_ends is None:
    x_ends = np.array([np.min(x), np.max(x)])
  if y_ends is None:
    y_ends = np.array([np.min(y), np.max(y)])

  # Mesh width
  dx = (x_ends[1] - x_ends[0]) / nx
  dy = (y_ends[1] - y_ends[0]) / ny
    
  # Create grid
  grid = np.zeros((n_avg, nx, ny))
  grid_count = np.zeros((n_avg, nx, ny))

  # Loop over frames
  for i, xi in enumerate(x):
    print('i = ', i)
    if centered:
      x_cm = np.mean(xi)
      y_cm = np.mean(y[i])
      xi -= x_cm 
      yi = y[i] - y_cm
    else:
      x_cm = 0
      y_cm = 0
      yi = y[i]

    # Shift frame values
    for j in range(n_avg - 1):
      grid[j,:,:] = grid[j+1,:,:]
      grid_count[j,:,:] = grid_count[j+1,:,:]
    grid[-1,:,:] = 0
    grid_count[-1,:,:] = 0

    # Loop over colloids
    for j in range(xi.size):
      rx = xi[j]
      ry = yi[j]
    
      # Position on grid
      nxj = nx - 1 - int((rx - x_ends[0]) / dx)
      nyj = ny - 1 - int((ry - y_ends[0]) / dy)

      # Save to 2d map
      if nxj >=0 and nxj < nx and nyj >= 0 and nyj < ny:
        grid[-1,nxj,nyj] += variable[i,j]
        grid_count[-1,nxj,nyj] += 1
        
    # Print png
    if file_prefix is not None:
      # Average grid over n_avg
      grid_sum = np.sum(grid, axis=0)
      grid_count_sum = np.sum(grid_count, axis=0)
      sel = grid_count_sum > 0
      grid_sum[sel] = grid_sum[sel] / grid_count_sum[sel]

      # Save to mesh
      alpha = np.zeros_like(grid_sum)
      alpha[sel] = 1
      x_center = np.linspace(x_ends[1] - dx / 2, x_ends[0] + dx / 2, num=nx)
      y_center = np.linspace(y_ends[1] - dy / 2, y_ends[0] + dy / 2, num=ny)
      mesh_x, mesh_y = np.meshgrid(x_center, y_center)
      
      # Plot
      name = file_prefix + '.step.' + str(i).zfill(8) + '.png'
      plt.clf()
      plot = plt.pcolormesh(mesh_y, mesh_x, grid_sum, cmap='viridis', alpha=alpha, vmin=vmin, vmax=vmax)
      plt.colorbar(plot)
      plt.xlabel('x ($\mu$m)')
      plt.ylabel('y ($\mu$m)')
      plt.savefig(name)


def cluster_detection_sklearn_optics(x, max_eps=np.inf, min_samples=5, n_save=1, file_prefix=None):
  '''
  Detect clusters using the sklearnb.cluster.OPTICS.
  '''
  # Prepare cluster
  clust = skcl.OPTICS(min_samples=min_samples, max_eps=max_eps, xi=0.05)
  
  # Loop over snapshots
  labels = []
  for i, xi in enumerate(x):
    print('snapshot = ', i)
    clust.fit(xi)
    
    # Get labels
    labels_i = skcl.cluster_optics_dbscan(reachability = clust.reachability_,
                                          core_distances = clust.core_distances_,
                                          ordering = clust.ordering_,
                                          eps=max_eps)
    labels.append(labels_i)

    if file_prefix and (i % n_save) == 0:
      # Prepare colors for a few curves
      C = plt.cm.viridis(np.linspace(0, 1, 5)) 
      
      # Create panel
      fig, axes = plt.subplots(1, 1, figsize=(5,5))

      axes.cla()
      # Loop over klasses
      for klass, color in enumerate(C):
        Xk = xi[labels_i == klass]       
        # Plot
        axes.scatter(Xk[:,0], Xk[:,1], color=color)
        
      # Plot non-clusters
      Xk = xi[labels == -1]    
      axes.scatter(Xk[:,0], Xk[:,1], color='grey', alpha=0.3)
  
      # Set axes and legend
      axes.set_xlabel('x')
      axes.set_ylabel('y')
      axes.legend()
      fig.tight_layout()

      # Save to pdf and png
      name = file_prefix + '.step.' + str(i).zfill(8) + '.clusters.png'
      plt.savefig(name, format='png') 

  return labels


def radius_of_gyration(r_vectors):
  '''
  Compute radius of gyration.

  It returns the squared radius of gyration tensor and the scalar radius of gyration
  '''
  r_cm = np.mean(r_vectors, axis=0)
  Rg_squared_ij = np.einsum('ni,nj->ij', r_vectors - r_cm, r_vectors - r_cm) / r_vectors.shape[0]
  Rg = np.sqrt(Rg_squared_ij[0,0] + Rg_squared_ij[1,1] + Rg_squared_ij[2,2])

  return Rg_squared_ij, Rg
