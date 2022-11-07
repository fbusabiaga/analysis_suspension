// Functions to write VTK files from C++ 
#include <boost/python.hpp>
#include <boost/python/numpy.hpp>
#include <iostream>
#include <math.h>
#include <stdio.h>
#include <vector>
#include "visit_writer.h"
#include "visit_writer.c"

namespace bp = boost::python;
namespace np = boost::python::numpy;

void visitWriterInterface(std::string name,
                          /*int format_file,*/
                          np::ndarray format_file,
                          np::ndarray dims,
                          np::ndarray x,
                          np::ndarray y,
                          np::ndarray z,
                          /*int nvars,*/
                          np::ndarray nvars,
                          np::ndarray vardims,
                          np::ndarray centering,
                          bp::list varnames,
                          bp::list variables){

  int *format_array = reinterpret_cast<int *>(format_file.get_data());
  int *nvars_array = reinterpret_cast<int *>(nvars.get_data());
  int *dims_array = reinterpret_cast<int *>(dims.get_data());
  int *vardims_array = reinterpret_cast<int *>(vardims.get_data());
  int *centering_array = reinterpret_cast<int *>(centering.get_data());

  // Copy names
  std::string varnames_array[nvars_array[0]];
  char **varnames_char = new char* [nvars_array[0]];
  char nameVar[nvars_array[0]][50]; 
  for(int i=0; i<nvars_array[0]; i++){
    varnames_array[i] = bp::extract<std::string>(varnames[i]);
    int sizeName = varnames_array[i].size();
    varnames_array[i].copy(nameVar[i], sizeName);
    varnames_char[i] = &nameVar[i][0];
  }

  // Copy variables
  double **vars;
  vars = new double* [nvars_array[0]];
  for(int i=0; i<nvars_array[0]; i++){
    np::ndarray variables_i = bp::extract<np::ndarray>(variables[i]);
    vars[i] = reinterpret_cast<double *>(variables_i.get_data());
  }

  // Copy mesh
  double* xmesh = reinterpret_cast<double *>(x.get_data());
  double* ymesh = reinterpret_cast<double *>(y.get_data());
  double* zmesh = reinterpret_cast<double *>(z.get_data());
  
  // Print variables
  if(0){
    std::cout << std::endl << "visitWriterInterface " << std::endl;
    std::cout << "name: " << name << std::endl;
    std::cout << "format: " << format_array[0] << std::endl;
    std::cout << "dims: " << dims_array[0] << "  " << dims_array[1] << "  " << dims_array[2] << std::endl;
    std::cout << "nvars: " << nvars_array[0] << std::endl;
    std::cout << "vardims: " << vardims_array[0] << std::endl;
    std::cout << "centering: " << centering_array[0] << std::endl;
    std::cout << "varnames: " << varnames_array[0] << std::endl;
    std::cout << std::endl;
  }

  // Call visit_writer
  /*Use visit_writer to write a regular mesh with data. */
  write_rectilinear_mesh(name.c_str(),    // Output file
                         format_array[0], // 0=ASCII,  1=Binary
                         dims_array,      // {mx, my, mz}
                         xmesh,           
                         ymesh,
                         zmesh,
                         nvars_array[0],  // number of variables
                         vardims_array,   // Size of each variable, 1=scalar, velocity=3*scalars
                         centering_array,  
                         varnames_char,   
                         vars);

  delete[] vars;
  delete[] varnames_char;
}


BOOST_PYTHON_MODULE(visit_writer_interface)
{
  using namespace boost::python;

  // Initialize numpy
  Py_Initialize();
  np::initialize();
  def("visit_writer_interface", visitWriterInterface);
}

