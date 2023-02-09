'''
Average velocity profiles and use them to extract the viscosity.
'''
import numpy as np
import sys
import glob
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy.optimize as scop

plt.rcParams.update({
  "text.usetex": True,
  "font.family": "DeJavu serif",
  "font.serif": ["Times New Roman"],
})

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
  file_prefix = '/home/fbalboa/simulations/RigidMultiblobsWall/rheology/data/run2000/run2133/run2133.3.rerun_1.' 
  suffix = '.velocity_profile.step.*.dat' 
  simulation_number_start = 0 
  simulation_number_end = 40 
  shift_linear_fit = 5 
  set_axis = 0
  dt_sample = 1e-05 * 25
  gamma_dot_0 = 1e+02
  eta_0 = 1e-03 
  wall_Lz = 4.5
  N_avg_fraction = np.array([1.1, 2])

  # Set output name
  output_name = file_prefix + '0-' + str(simulation_number_end) + '.velocity_profile.step.average.dat'

  # Create one panel
  fig, axes = plt.subplots(1, 1, figsize=(5,5))

  # Get files
  files = []
  for i in range(simulation_number_start, simulation_number_end + 1):
    name = file_prefix + str(i) + suffix   
    files_i = [filename for filename in glob.glob(name)]
    files_i.sort()
    files_i = files_i[0:-1]
    files.extend(files_i)
    
  # Read files
  N_files = len(files)
  x = []
  for i in range(len(files)):
    name = files[i]
    print('i = ', i, name)
    x.append(np.loadtxt(name))
  x = np.asarray(x)

  # Set number of average curves
  N_avg = (N_files / N_avg_fraction).astype(int)
  C = mpl.cm.viridis(np.linspace(0, 1, N_avg.size + 1))

  # Loop
  for k, N_avg_i in enumerate(N_avg):
    def linear_profile(y, offset, slope):
      return offset + slope * y

    # Prepare variables
    viscosity = []
    viscosity_error = []
    time = []
    shear_rate = []

    # Loop  over segments
    for i in range(0, N_files - N_avg_i):
      xi = x[i : i + N_avg_i]
      x_avg = np.mean(xi, axis=0)
      time.append(dt_sample * (2*i + N_avg_i - 1) / 2)
      
      # Fit profile
      try:
        popt, pcov = scop.curve_fit(linear_profile, x_avg[shift_linear_fit:-shift_linear_fit,0], x_avg[shift_linear_fit:-shift_linear_fit,set_axis+1])
      except:
        print('error fitting')
        sys.exit()
        popt = np.ones(2)
        pcov = np.ones((2,2))
        
      # Set viscosity
      shear_rate.append(popt[1])
      viscosity.append(eta_0 * gamma_dot_0 / popt[1])
      viscosity_error.append(eta_0 * gamma_dot_0 * np.sqrt(pcov[1,1]) / popt[1]**2)
    
    # Plot viscosity
    print('Making plot, N_avg_i = ', N_avg_i)
    time = np.array(time)
    viscosity = np.array(viscosity)
    viscosity_error = np.array(viscosity_error)
    axes.plot(gamma_dot_0 * time, viscosity / eta_0, '-', color=C[k])

  # Set axes
  axes.set_xlabel(r'$\dot{\gamma}_0 t$', fontsize=fontsize)
  axes.set_ylabel(r'$\eta\, /\, \eta_0$', fontsize=fontsize)
  # axes.set_ylim(0, 200)
  axes.tick_params(axis='both', which='major', labelsize=fontsize)
  axes.yaxis.offsetText.set_fontsize(fontsize)
  axes.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
       
  # Adjust distance between subplots
  fig.tight_layout()
  # fig.subplots_adjust(left=0.13, top=0.95, right=0.9, bottom=0.17, wspace=0.0, hspace=0.0)
  
  # Save to pdf and png
  plt.savefig('plot_viscosity_vs_time.py.pdf', format='pdf') 
  name = file_prefix + 'base.plot_viscosity_vs_time.py.pdf'
  plt.savefig(name, format='pdf') 
    
