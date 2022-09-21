'''
Average velocity profiles and use them to extract the viscosity.
'''
import numpy as np
import sys
import glob
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy.optimize as scop

if True:   
  # Set some global options
  fontsize = 18
  mpl.rcParams['axes.linewidth'] = 1.5
  mpl.rcParams['xtick.direction'] = 'in'
  mpl.rcParams['xtick.major.size'] = 4
  mpl.rcParams['xtick.major.width'] = 1.5
  mpl.rcParams['xtick.minor.size'] = 4
  mpl.rcParams['xtick.minor.width'] = 1
  mpl.rcParams['ytick.direction'] = 'in'
  mpl.rcParams['ytick.major.size'] = 4
  mpl.rcParams['ytick.major.width'] = 1.5
  mpl.rcParams['ytick.minor.size'] = 4
  mpl.rcParams['ytick.minor.width'] = 1


if __name__ == '__main__':
  file_prefix = '/home/fbalboa/simulations/RigidMultiblobsWall/rheology/data/run2000/run2133/run2133.6.rerun_0.'
  suffix = '.velocity_profile.step.*.dat'
  simulation_number_start = 0
  simulation_number_end = 1
  N_skip_fraction = 4
  N_intervales = 3
  num_columns = 4
  shift_linear_fit = 10
  set_axis = 0
  gamma_dot_0 = 1e+04
  eta_0 = 1e-03
  wall_Lz = 4.5

  # Set output name
  output_name = file_prefix + '0-' + str(simulation_number_end) + '.velocity_profile.step.average.dat'

  # Create one panel
  fig, axes = plt.subplots(1, 1, figsize=(5,5))
  
  # Prepare colors for a few curves
  C = mpl.cm.Greys(np.linspace(0, 1, N_intervales + 1))


  # Get files
  files = []
  for i in range(simulation_number_start, simulation_number_end + 1):
    name = file_prefix + str(i) + suffix
    files_i = [filename for filename in glob.glob(name)]
    files_i.sort()
    files_i = files_i[0:-1]
    files.extend(files_i)


  # Read files
  N_skip = len(files) // N_skip_fraction
  x = []
  for i in range(N_skip, len(files)):
    name = files[i]
    print('i = ', i, name)
    x.append(np.loadtxt(name))
  x = np.asarray(x)
    
  # Compute mean and ste
  x_avg = np.mean(x, axis=0)
  x_ste = np.std(x, axis=0, ddof=1) / np.sqrt(len(files) - N_skip)

  # Save result
  result = np.zeros((x_avg.shape[0], x_avg.shape[1] * 2))
  result[:,0:x_avg.shape[1]] = x_avg
  result[:,x_avg.shape[1]:] = x_ste
  np.savetxt(output_name, result)
  

  if True:
    def linear_profile(y, offset, slope):
      return offset + slope * y

    # Prepare variables
    v0 = np.zeros(N_intervales)
    slope = np.zeros(N_intervales)
    slope_error = np.zeros(N_intervales)
    eta = np.zeros(N_intervales)
    eta_error = np.zeros(N_intervales)

    # Loop  over segments
    len_segment = x.shape[0] // N_intervales
    for i in range(N_intervales):
      xi = x[i*len_segment : (i+1)*len_segment]
      x_avg = np.mean(xi, axis=0)
      
      # Fit profile
      try:
        if shift_linear_fit > 0:
          axes.plot(x_avg[:,0], x_avg[:,set_axis+1], '-', color=C[i])
          popt, pcov = scop.curve_fit(linear_profile, x_avg[shift_linear_fit:-shift_linear_fit,0], x_avg[shift_linear_fit:-shift_linear_fit,set_axis+1], p0=[-3800, 1700])          
        else:
          popt, pcov = scop.curve_fit(linear_profile, x_avg[:,0], x_avg[:,set_axis+1], p0=[0.1, 1])
      except:
        print('error fitting')
        popt = np.ones(2)
        pcov = np.ones((2,2))
        
      # Set viscosity
      v0[i] = popt[0]
      slope[i] = popt[1]
      eta[i] = eta_0 * gamma_dot_0 / popt[1]
      eta_error[i] = eta_0 * gamma_dot_0 * np.sqrt(pcov[1,1]) / popt[1]**2
      slope_error[i] = np.sqrt(pcov[1,1])
      print('i       = ', i, ', eta = ', eta[i] / eta_0, ' +/- ', eta_error[i] / eta_0)


    # Average viscosity
    eta_avg = np.mean(eta)
    eta_std = np.std(eta, ddof=1) / np.sqrt(N_intervales) + np.mean(eta_error)
    slope_avg = np.mean(slope)
    slope_std = np.std(slope, ddof=1) + np.mean(slope_error)
    print('eta_avg = ', eta_avg / eta_0, ' +/- ', eta_std / eta_0)
    print('eta_avg = ', eta_avg, eta_std)
    print(' ')
    print('v0      = ', np.mean(v0) + np.mean(slope) * wall_Lz / 2, ' +/- ', np.std(v0, ddof=1) + np.std(slope) * wall_Lz / 2 / np.std(slope)**2)
    print(' ')
    
    # Plot average
    v0 = np.mean(v0)
    axes.plot(x[0,:,0], linear_profile(x[0,:,0], v0, eta_0 * gamma_dot_0 / eta_avg), '--', color='r')

    # Set axes
    axes.set_xlabel(r'$z$ ($\mu$m)', fontsize=fontsize)
    axes.set_ylabel(r'$v(z)$ ($\mu$m/s)', fontsize=fontsize)
    axes.tick_params(axis='both', which='major', labelsize=fontsize)
    axes.yaxis.offsetText.set_fontsize(fontsize)
    axes.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
       
    # Adjust distance between subplots
    fig.tight_layout()
    # fig.subplots_adjust(left=0.13, top=0.95, right=0.9, bottom=0.17, wspace=0.0, hspace=0.0)
  
    # Save to pdf and png
    plt.savefig('plot_velocity_profiles.py.pdf', format='pdf') 
    name = file_prefix + 'plot_velocity_profiles.py.pdf'
    plt.savefig(name, format='pdf') 
    
