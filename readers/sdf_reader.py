#!/usr/local/bin/python

import numpy as np
from scipy import constants as sp
import sdf
import os
import imp
import data_reader
import utils.postproc_tools as tools
import subprocess

class SDF_Reader(data_reader.Data_Reader):
	"""Data reader for EPOCH. SDF_Reader reads .sdf files"""
	def __init__(self, **kwargs):		
		path = kwargs.get('path')
		self.run = kwargs.get('run')
		self.si = kwargs.get('si', False)
		# determine if data_reader displays information to stdout
		self._quiet = kwargs.get('quiet', False)
		# a list of names used to search for the maximum density in the input.deck 
		# so that data can be normalized to dimensionless units
		self.density_names = ['ne', 'Ne', 'n_max', 'rho_max']
		# if one of the density names cannot be found, si units must be used
		self._require_si = False
		self._lambda_L = kwargs.get('lambda_L', 0.8e-6)
		# determines if k0 should be used for normalizations. If False, then kp is used
		self._use_k0 = kwargs.get('use_k0', False)
		# used to return data within the given x and y ranges
		self.extent_x = kwargs.get('extent_x', ())
		self.extent_y = kwargs.get('extent_y', ())
		# if True, the extent values are given in SI units
		self._extent_si = kwargs.get('extent_si', False)
		# if True, the extent values have been scaled by the laser wavenumber k0
		self._extent_k0 = kwargs.get('extent_k0', False)
		# the x coordinate being used: x or xi
		self._xcoord = kwargs.get('xcoord', 'x')
		# dictionary to hold reference values such as w, wp, kp, w0, lambda_L, etc...
		self._reference = {}	

		# set alternate names that can be used when specifying data types
		self._alt_names = {
			'x': 'x grid coordinate',
			'y': 'y grid coordinate',
			'z': 'z grid coordinate',
			'Ex': 'Electric Field/Ex',
			'Ey': 'Electric Field/Ey',
			'Ez': 'Electric Field/Ez',
			'Bx': 'Magnetic Field/Bx',
			'By': 'Magnetic Field/By',
			'Bz': 'Magnetic Field/Bz',
			'Jx': 'Current/Jx',
			'Jy': 'Current/Jy',
			'Jz': 'Current/Jz',
			'Px': 'Particles/Px/electron',
			'Py': 'Particles/Py/electron',
			'Pz': 'Particles/Pz/electron',
			'id': 'Particles/ID/electron',
			'HePx': 'Particles/Px/Helium',
			'HePy': 'Particles/Py/Helium',
			'HePz': 'Particles/Pz/Helium',
			'Heid': 'Particles/ID/Helium',
			'weight': 'Particles/Weight/electron',
			'Heweight': 'Particles/Weight/Helium',
			'ppc': 'Particles/Particles Per Cell/electron',
			'Heppc': 'Particles/Particles Per Cell/Helium',
			'pos_x': 'x coordinate of electrons',
			'pos_x': 'moving window coordinate of electrons',
			'pos_y': 'y coordinate of electrons',
			'pos_z': 'z coordinate of electrons',
			'Hepos_x': 'x coordinate of Heliums',
			'Hepos_y': 'y coordinate of Heliums',
			'Hepos_z': 'x coordinate of Heliums',
			'num_e': 'total number of electrons in the simulation',
			'Grid': 'Grid/Grid',
			'Ne': 'Derived/Number_Density/electron',
			'NHe': 'Derived/Number_Density/Helium',
			'N': 'Derived/Number_Density',
			'dt': 'Time increment',
			'time': 'simulation time of EPOCH'
		}

		# inputs that need to be indexed to get the proper direction
		self._directional = {
			'x': 'Grid/Grid',
			'y': 'Grid/Grid',
			'z': 'Grid/Grid',
			'pos_x': 'Grid/Particles/electron',
			'pos_y': 'Grid/Particles/electron',
			'pos_z': 'Grid/Particles/electron',
			'Hepos_x': 'Grid/Particles/Helium',
			'Hepos_y': 'Grid/Particles/Helium',
			'Hepos_z': 'Grid/Particles/Helium'
		}

		# determine dump information
		self._time_range = kwargs.get('time')
		self._dump_range = kwargs.get('dump')		

		# set path
		self.run_data = None
		self._set_path(path)
		

		# determine how many timesteps are being used
		self.num_timesteps = self.last - self.first + 1
		# boolean used to determine if it is the first time data is read
		self._is_first_read = True
		self._first_read()

		# define the function for getting indices of an array for extent_x
		if self.extent_x == 'all':
			self.get_ix = self._get_ix_all
			self._set_extent_x_max()
		elif self.extent_x:
			self.get_ix = self._get_ix
		else:
			self.get_ix = self._get_ix_all
			self._set_extent_x_max()

		# define the function for getting indices of an array for extent_y
		if self.extent_y == 'all':
			self.get_iy = self._get_iy_all
			self._set_extent_y_max()
		elif self.extent_y:
			self.get_iy = self._get_iy
		else:			
			self.get_iy = self._get_iy_all
			self._set_extent_y_max()

	def _set_extent_x_max(self):
		x = self.get(self._xcoord, dump=self._current_sdf)
		self.extent_x = (np.min(x), np.max(x))

	def _set_extent_y_max(self):
		y = self.get('y', dump=self._current_sdf)
		self.extent_y = (np.min(y), np.max(y))

	def _get_ix(self, x, **kwargs):
		"""Return the indices in which <x> is within extent_x."""
		ix, = np.where((x > self.extent_x[0]) & (x < self.extent_x[1]))
		return ix

	def _get_ix_all(self, x, **kwargs):
		"""Return the indices in which <x> is within extent_x."""
		self.extent_x = (np.min(x), np.max(x))
		return np.linspace(0, len(x)-1, len(x), dtype=int)

	def _get_iy(self, y, **kwargs):
		"""Return the indices in which <y> is within extent_y."""
		iy, = np.where((y > self.extent_y[0]) & (y < self.extent_y[1]))
		return iy 

	def _get_iy_all(self, y, **kwargs):
		"""Return the indices in which <y> is within extent_y."""
		self.extent_y = (np.min(y), np.max(y))
		return np.linspace(0, len(y)-1, len(y), dtype=int)  
	
	def _get_string(self, sdf_num):
		"""Pass a number of 4 digits or less to be turned into an sdf string"""
		string = str(sdf_num)
		while len(string) < 4: # .sdf files need to have four digits. pad with zeros
			string = '0' + string
		return string

	def _get_number(self, sdf_num): 
		"""Pass a list of 4 characters to be turned into sdf number.
		If SDF_NUM ends in '.sdf' then it will be removed before determining number"""
		sdf_num = sdf_num.replace('.sdf', '')
		sdf_num = list(sdf_num)
		n = len(sdf_num)
		# if the run size is large enough, there can be more digits in the .sdf than 4
		if n > 5: # if it still has the .sdf attached
			del sdf_num[n-4:n] # delete the .sdf 
		while sdf_num[0] == '0' and len(sdf_num) > 1:
			del sdf_num[0]
		num_string = "".join(sdf_num)
		return int(num_string)

	def sdf_valid(self, sdf_num):
		"""Check if a sdf file is corrupted or in the process of being written by EPOCH."""
		try: sdf.read(str(self.path + self._get_string(sdf_num) + '.sdf'), dict=True)['Grid/Grid'].data
		except: return False
		else: return True

	def _find_first_last(self):
		"""Finds the first and last valid sdf in directory given by path"""
		min_sdf = np.inf
		max_sdf = -1
		for file in os.listdir(self.path):
			if file.endswith('.sdf'):
				temp = self._get_number(file)
				if temp > max_sdf:
					max_sdf = temp
				if temp < min_sdf:
					min_sdf = temp
		if max_sdf != -1 and min_sdf != np.inf:
			# ensure the last sdf is valid
			for i in range(max_sdf, min_sdf-1, -1):
				if self.sdf_valid(i) is True:
					max_sdf = i
					break
			else:
				raise OSError('No .sdf files exist in %s' % self.path)
			# ensure the first sdf is valid
			for i in range(min_sdf, max_sdf+1):
				if self.sdf_valid(i) is True:
					min_sdf = i
					break
			else:
				raise OSError('No .sdf files exist in %s' % self.path)
			return min_sdf, max_sdf

		# no sdf files found, raise error
		raise OSError('No .sdf files exist in %s' % self.path)	

	def _get_num_dumps(self, absolute=False):
		"""Get the number of sdf files in the directory.
		Kwargs:
		  absolute: If True, give the total number of .sdf files regardless of an adjusted time/dump range
		"""
		if absolute is True:
			first = self.abs_first
			last = self.abs_last
		else:
			first = self.first
			last = self.last
		num_dumps = 0
		for file in os.listdir(self.path):
			if file.endswith('.sdf'):
				num = self._get_number(file)
				if num >= first and num <= last:
					num_dumps += 1
		return num_dumps

	def _get_sdf_list(self):
		"""Returns a list of all data types outputted to the SDF files"""
		temp_list = []
		for name, data in self._current_dict.items():
			temp_list.append(name)
		return sorted(temp_list)

	def _first_read(self):
		if self._is_first_read is True and self._quiet is False:
			# let the user know where the data is being read from
			print 'reading data from: %s' % self.path
			self._is_first_read = False

	def update(self):
		"""If the directory the reader is reading from has changed, update the reader's information"""
		self.abs_first, self.abs_last = self._find_first_last()
		self.first, self.last = self.abs_first, self.abs_last		

		# set the current SDF to the first SDF
		self._current_sdf = self.first		
		self._current_dict = sdf.read(str(self.path + self._get_string(self._current_sdf) + '.sdf'), dict=True)		

		# get a list of data types in the .sdf files
		self.sdf_data_types = self._get_sdf_list()

		# code_name is Epoch#d and dim is an integer (1,2,3)
		self.code_name = self.get('Header')['code_name']
		self.dim = int(self.code_name.lower().split('epoch', 1)[1][0])

		# determine if the code used a moving window
		self._moving_window = self.get_input('move_window', val_only=True, silent=True)
		if self._moving_window == 'T':
			self._moving_window = True
		else:
			self._moving_window = False

		# get the density from the input.deck
		try: n0 = self._get_variable(['ne', 'Ne', 'n_max', 'rho_max'], si=True)
		# if vars.py doesn't exist, try reading from the input itself
		except IOError as e:
			if self._quiet is False:
				print 'failed to create vars.py'
				print 'reading density from input.deck...'
			n0 = self._get_density()
		# set normalization values
		if n0:
			w0 = 2. * sp.pi * sp.c / self._lambda_L
			k0 = w0 / sp.c
			wp = np.sqrt((n0 * sp.e**2) / (sp.m_e * sp.epsilon_0))
			kp = wp / sp.c
			w = w0 / wp			
			
			# set k_kp and kp
			self.k_kp = k0 / kp
			self.kp = kp

			# normalize based on k0 or kp
			if self._use_k0:
				E0 = sp.m_e * w0 * sp.c / sp.e
				# set the normalization of the time and space variables to k0
				self._space_norm = k0
				self._time_norm = w0
			else:
				E0 = sp.m_e * wp * sp.c / sp.e
				# set the normalization of the time and space variables to kp
				self._space_norm = kp
				self._time_norm = wp

			B0 = E0 / sp.c
			P0 = sp.m_e * sp.c
			J0 = sp.c * sp.e * n0

			self._reference = {
				'n0': n0,
				'lambda_L': self._lambda_L,
				'w0': w0,
				'k0': k0,
				'wp': wp,
				'kp': kp,							
				'w' : w,				
				'E0': E0,
				'B0': B0,
				'P0': P0,
				'J0': J0
			}

			# create a dictionary that contains normalization factors for conversion to dimensionless units
			self._normalize = {
				'Electric Field/Ex'              : 1. / E0,
				'Electric Field/Ey'              : 1. / E0,
				'Electric Field/Ez'              : 1. / E0,
				'Magnetic Field/Bx'              : 1. / B0,
				'Magnetic Field/By'              : 1. / B0,
				'Magnetic Field/Bz'              : 1. / B0,
				'Current/Jx'                     : 1. / J0,
				'Current/Jy'                     : 1. / J0,
				'Current/Jz'                     : 1. / J0,
				'Derived/Number_Density/electron': 1. / n0,
				'Derived/Number_Density'         : 1. / n0,
				'Derived/Number_Density/Helium'  : 1. / n0,					
				'Particles/Px/electron'          : 1. / P0,
				'Particles/Py/electron'          : 1. / P0,
				'Particles/Pz/electron'          : 1. / P0,
				'Grid/Particles/electron'        : self._space_norm,
				'Grid/Grid'                      : self._space_norm,
				'spot_size'                      : self._space_norm,
				'rayleigh_length'                : self._space_norm,
				'dx'                             : self._space_norm,
				'dy'                             : self._space_norm,
				'dz'                             : self._space_norm,
				'time'                           : self._time_norm,
				'Time increment'                 : self._time_norm				
			}

			# determine if extents are given as si units
			# if so, convert si extents to dimless
			if self._extent_si is True and self.si is False:
				if self.extent_x:
					self.extent_x = (self.extent_x[0]/kp, self.extent_x[1]/kp)
				if self.extent_y:
					self.extent_y = (self.extent_y[0]/kp, self.extent_y[1]/kp)

			# determine if extents are given in terms of k0
			# if so, scale them to be in terms of kp
			if self._extent_k0 is True and self.si is False:
				kp_k0 = kp / k0
				print 'kp_k0: {}'.format(kp_k0)
				if self.extent_x:
					self.extent_x = (self.extent_x[0]*kp_k0, self.extent_x[1]*kp_k0)
				if self.extent_y:
					self.extent_y = (self.extent_y[0]*kp_k0, self.extent_y[1]*kp_k0)

		# could not find density to normalize. Using S.I.
		else:
			if self._quiet is False:
				print 'sdf_reader: could not read maximum density. Using S.I. units.'
			self.si = True
			self._require_si = True

		# adjust self.first and self.last based on a given time/dump range
		if self._time_range: 
			self.set_time_range(self._time_range)
		elif self._dump_range:
			self.set_time_range(self._dump_range, isdump=True)

		# self.first may have changed during set_time_range
		self._current_sdf = self.first		
		self._current_dict = sdf.read(str(self.path + self._get_string(self._current_sdf) + '.sdf'), dict=True)

		# number of SDF files
		self.num_dumps = self._get_num_dumps()
		
		# set the number of cells in each direction
		self._get_num_cells()

		# set max position values
		self.xmax = self.get('x', dump=self.last)[-1]
		self.xmin = self.get('x', dump=self.first)[0]
		self.ymax = self.get('y', dump=self.last)[-1]
		self.ymin = self.get('y', dump=self.first)[0]

	def _set_path(self, new_path):
		"""Defines a new path and sets up information about the directory"""
		# if path is none use default
		if new_path is None:
			new_path = os.environ['epoch2d'] + '/data'
			# ensure path exists
			if new_path is None:
				errmsg = 'no path given to SDF_reader. Give SDF_reader(path=desired_path) or set default_dir'
				raise IOError(errmsg)

		# check if new_path is a shortcut for a data dir in fuchs_data
		# i.e., dir=8-1 as opposed to dir=run_8/data_1		
		if "/" not in new_path and "-" in new_path:
			temp = new_path.split('-')
			run = temp[0]
			data = temp[1]
			self.run_data = (new_path, run, data)
			new_path = os.environ['run_data'] + '/run_{}/data_{}'.format(run, data)

		# check if a run number has been given, if so use that directory
		if self.run:
			# convert to string and pad with 0s if need by
			self.run = str(self.run)
			while len(self.run) < 5:
				self.run = '0{}'.format(self.run)
			print 'path: {}'.format(tools.get_data_path())
			new_path = os.path.join(tools.get_data_path(), self.run)

		# ensure the path exists
		if os.path.isdir(new_path) is False:
			errmsg = 'directory does not exist: %s' % new_path
			raise OSError(errmsg)

		# new_path must end with a '/' for appending .sdf file names
		if not new_path.endswith('/'):
			new_path = new_path + '/'

		self.path = new_path
		self.update()

	def _get_variable(self, var_name, **kwargs):
		"""Get a variable from vars.py created from the EPOCH stub program.
		var_name is either a string or a list of strings.
		"""
		si = kwargs.get('si', self.si)
		# ensure var_name is a list
		if not isinstance(var_name, list):
			var_name = [var_name]
		# set the path for the variable file
		path = os.path.join(self.path, 'vars.py')

		# if vars.py does not exist, attempt to create it
		if not os.path.exists(path):
			# redirect output of stub program to /dev/null
			FNULL = open(os.devnull, 'w')
			if self._quiet is False:
				print 'calling EPOCH stub program to create vars.py...'
			try: subprocess.call(['getvars', self.path], stdin=FNULL, stdout=FNULL, stderr=FNULL)
			except: raise IOError('failed to create vars.py')
			finally: FNULL.close()

		# import variables
		v = imp.load_source('vars', path)

		# search for the variable name
		for name in var_name:
			try: val = getattr(v, name)
			except AttributeError: pass
			else: break
		else:
			raise AttributeError('could not find variable from vars.py: {}'.format(', '.join(var_name)))

		# convert to dimless is necessary
		if si is False:
			val *= self._normalize['spot_size']
		return val

	# allows the user to get a value(s) from the input.deck
	# returns a tuple of (param_name, value)
	# param name is the name of the parameter from the input.deck
	# *args: multiple parameters can be given at once. A list will be returned if this is the case
	# **kwargs: (param_name = alt_param_name) can be given and the returned list
	# will replace param_name with alt_param_name
	def get_input(self, param_name, *args, **kwargs):
		silent = kwargs.get('silent', self._quiet)
		val_only = kwargs.get('val_only', False)
		# check to see if there are more than one inputs to get
		if len(args) > 0:
			param_list = []
			param_list.append(self.get_input(param_name, **kwargs))
			for arg in args:
				param_list.append(self.get_input(arg, **kwargs))
			return param_list
		# if only one input is given, attempt to find its value
		else:
			try:
				input_path = os.path.join(self.path, "input.deck")
				file = open(input_path, 'r')
				for line in file:
					line = line.strip()
					if line.startswith(param_name):
						name, remainder = line.split('=', 1)
						# ensure param_name matches exactly
						if param_name == name.strip():
							# remove any comments
							val = remainder.split('#', 1)[0].strip()
							# check to see if an alternate name was given
							for arg in kwargs:
								if param_name == arg:
									return (kwargs[arg], val)
							if val_only is True:
								return val
							else:
								return (param_name, val)

				# if the input is not found let the user know
				if silent is False:
					print '{} was not found in: {}'.format(param_name, input_path)
			except IOError:
				errmsg = 'Error reading input.deck'
				raise IOError(errmsg)	

	# find the density from the input.deck and return its value
	def _get_density(self, **kwargs):
		density_name = kwargs.get('name')
		if density_name is None:
			def check_name(name, silent):
				try:
					temp, n0_str = self.get_input(name, silent=silent)
				except:
					return False
				else:
					return True

			self.density_names = ['ne', 'Ne', 'n_max', 'rho_max']
			names_str = ''
			for name in self.density_names:
				# build names_string
				names_str = names_str + ' "'+name+'"'
				if check_name(name, True) is True:
					return self._get_density(name=name)

			# # name not in default names. get user input
			# found_name = False
			# while (found_name is False):
			# 	input_str = input_str = "enter name for density in input.deck (i.e.%s): " % names_str
			# 	try_name = raw_input(input_str)
			# 	found_name = check_name(try_name, False)
			# # found name
			# return self._get_density(name=try_name)

			# density not found
			return None
			

		# name given
		else:
			# example input: n_max = 5e18 / cc # this is the max density
			temp, n0_str = self.get_input(density_name, silent=False)
			n0_list = n0_str.split('/', 1)
			# get the base value. i.e, 5e18
			n0_val = n0_list[0].strip()
			n0_val = float(n0_val)
			# get a modifier if it exists. i.e, 5e18 / cc
			if len(n0_list) == 2:
				mod = n0_list[1].strip()
				if mod == 'cc':
					n0_val /= 1e-6
			return n0_val

	# checks to see if a data type is being outputted to the SDF files
	# if exit=True, then the program will be exitted if the data type does not exist
	def _data_type_exists(self, data_type, exit):
		for t in self.sdf_data_types:
			if data_type == t:
				return True
		if exit:
			raise IOError('%s not given as output in SDF file' % data_type)
		else:
			return False

	def to_si(self, data_type, data):
		"""Convert data from si to dimless"""
		if self._require_si is True:
			print 'ERROR: could not convert from S.I. to dimless. No normalization constants could be found'
			return
		try:
			conversion = self._normalize[data_type]
		except KeyError:
			print "ERROR: could not convert from S.I. to dimless. '{}' is not a valid data type. Try 'ppt show'".format(data_type)
		else:
			print '(S.I. -> dimless) {}: {} -> {}'.format(data_type, data, data / conversion)

	def to_dimless(self, data_type, data):
		"""Convert data from si to dimless"""
		if self._require_si is True:
			print 'ERROR: could not convert from dimless to S.I. No normalization constants could be found'
			return
		try:
			conversion = self._normalize[data_type]
		except KeyError:
			print "ERROR: could not convert from dimless to S.I. '{}' is not a valid data type. Try 'ppt show'".format(data_type)
		else:
			print '(dimless -> S.I.) {}: {} -> {}'.format(data_type, data, data * conversion)

	# some inputs such as Grid/Grid return a multidimensional array for each dimension of the grid (x,y,z)
	# get(data_type) will call get_directional if data_type is a directional input
	def _get_directional(self, data_type):
		# get directional data
		if len(data_type) == 1:
			dir_data = data_type
		else:
			dir_data = data_type.split('_', 1)[1]

		# return the appropriate index or error
		if dir_data == 'x':
			return 0
		elif dir_data == 'y':
			return 1
		else:
			return 2

	# returns a tuple containing the grid size in each direction
	# (num_x, num_y, num_z)
	def _get_num_cells(self):
		grid = self.get('Grid')
		num_x = len(grid[0]) - 1

		# set 1d cell values
		self.num_cells_x = num_x
		self.num_cells_y = None
		self.num_cells_z = None

		# set 2d cell values
		if self.dim > 1:
			self.num_cells_y = len(grid[1]) - 1
			# set 3d cell values
			if self.dim > 2:
				self.num_cells_z = len(grid[2]) - 1

	def has_dump(self, dump):
		"""Return true if DUMP exists, false otherwise."""
		try:
			sdf.read(str(self.path + self._get_string(dump) + '.sdf'), dict=True)
		except IOError:
			return False
		return True

	def sdf_exists(self, sdf_num):
		"""Check if a SDF file exists"""
		sdf_name = '{}.sdf'.format(self._get_string(sdf_num))
		for file in os.listdir(self.path):
			if file == sdf_name:
				return True
		return False

	def get_dump(self, time, tol=0.1, ignore=False):
		"""Return the dump number that is nearest to time.
		Kwargs:
		  tol: the tolerance -- determined by percent error -- in which the found index's value must match
			ignore: If True, then ignore the tolerance check
		"""
		if time is None: return
		dump = 0
		nearest = np.inf
		t_nearest = np.inf
		for i in range(self.first, self.last+1):
			if self.has_dump(i) is False: continue
			t = self.get('time', dump=i)
			diff = abs(time - t)
			if diff < nearest:
				dump = i
				nearest = diff
				t_nearest = t
				
		# hack to calculate the percent error of a value of 0
		# alternatives should probably be looked into
		if time == 0.0:
			perc_err = ((1. + np.abs(t_nearest - time)) / 1.) - 1.
		else:	
			perc_err = np.abs(t_nearest - time) / time
		if perc_err > tol and ignore is False:
			raise ValueError("could not find dump for time '{}' within tolerance: {}".format(time, tol))
		return dump

	def _get_xi(self, **kwargs):
		"""Return the longitudinal position in moving window coordinates (xi=ct-x)."""
		if self._moving_window is False:
			return self.get('x', **kwargs)

		si = kwargs.get('si', self.si)
		if self._require_si is True: si = True
		sdf_num = kwargs.get('dump')
		xi = (sp.c * self.get('time', dump=sdf_num, si=True)) - self.get('x', dump=sdf_num, si=True)
		if not si:
			try:
				xi = np.multiply(xi, self._space_norm)
			except KeyError:
				raise KeyError("Could not find value for 'n0'. No normalizations can be made")
		return xi

	def _get_pos_xi(self, **kwargs):
		"""Return the longitudinal particle position in moving window coordinates (xi=ct-x)."""
		if self._moving_window is False:
			return self.get('pos_x', **kwargs)

		si = kwargs.get('si', self.si)
		if self._require_si is True: si = True
		sdf_num = kwargs.get('dump')
		xi = (sp.c * self.get('time', dump=sdf_num, si=True)) - self.get('pos_x', dump=sdf_num, si=True)
		if not si:
			try:
				xi = np.multiply(xi, self._space_norm)
			except KeyError:
				raise KeyError("Could not find value for 'n0'. No normalizations can be made")
		return xi

	def _get_cell_size(self, data_type, **kwargs):
		"""Calculates the cell size if they aren't given in the output."""
		si = kwargs.get('si', self.si)
		if self._require_si is True: si = True
		# if data_type doesn't exit, _data_type_exists() will exit the program
		exists = self._data_type_exists(data_type, False)
		
		if exists is True:
			data = self._current_dict[data_type].data			
		# find cell size from the grid itself
		else:
			if data_type == 'dx':
				grid = self.get('x', si=True)
			elif data_type == 'dy':
				grid = self.get('y', si=True)
			else:
				grid = self.get('z', si=True)
			data = grid[1] - grid[0]

		# return a dimless value is kwarg is given as true
		if not si:
			norm = self._normalize.get(data_type, 1)
			data = np.multiply(data, norm)
		return data			

	def get_cell_center(self, coord, **kwargs):
		"""Given a coordinate <coord>, return the cell center for self.get(<a>).
		The returned list will be len(<a>) - 1."""
		si = kwargs.get('si', self.si)
		t = kwargs.get('dump', 0)
		# get the offset = cell_size / 2
		if coord == 'x':
			offset = self._get_cell_size('dx', si=True) / 2.
			a = self.get('x', dump=t, si=True)
		elif coord == 'y':
			offset = self._get_cell_size('dy', si=True) / 2.
			a = self.get('y', dump=t, si=True)
		elif coord == 'z':
			offset = self._get_cell_size('dz', si=True) / 2.
			a = self.get('z', dump=t, si=True)
		elif coord == 'xi':
			offset = self._get_cell_size('dx', si=True) / 2.
			a = self.get('xi', dump=t, si=True)
		else:
			raise ValueError('Not a valid coordinate. Try: x, y, z, or xi')
		# calculate the cell center
		center = np.empty(a.size - 1)
		for i in range(a.size - 1):
			center[i] = a[i] + offset
		if not si:
			try:
				center = np.multiply(center, self._space_norm)
			except KeyError:
				raise KeyError("Could not find value for 'n0'. No normalizations can be made")
		return center

	# set the current sdf and sdf_dict
	def _set_current(self, sdf_num):
		if self._current_sdf != sdf_num:
			self._current_sdf = sdf_num
			self._current_dict = sdf.read(str(self.path + self._get_string(self._current_sdf) + '.sdf'), dict=True)

	# returns data given the name of the data type and the sdf number to read from
	# **kwargs: (sdf = sdf_num) an integer which determines which .SDF file to read data from
	# if no sdf is given, then the default is the first .SDF file
	# time overrides dump
	def get(self, data_type, **kwargs):
		# data is dimless by default, si=True to return S.I. units
		si = kwargs.get('si', self.si)
		if self._require_si is True: si = True
		time = kwargs.get('time')
		# dump overrides time
		if time:
			sdf_num = kwargs.get('dump', self.get_dump(time))
		else:
			sdf_num = kwargs.get('dump', self._current_sdf)
		# set the sdf to read from
		self._set_current(sdf_num)

		# special cases
		if data_type == 'num_cells_x':
			return self.num_cells_x
		elif data_type == 'num_cells_y':
			return self.num_cells_y
		elif data_type == 'num_cells_z':
			return self.num_cells_z
		elif data_type in self._reference:
			return self._reference[data_type]
		elif data_type == 'xi':
			return self._get_xi(**kwargs)
		elif data_type == 'pos_xi':
			return self._get_pos_xi(**kwargs)
		elif data_type in ['num_e', 'num_particles'] :
			return self.get('pos_x', dump=sdf_num, si=True).size
		elif data_type in ['dx', 'dy', 'dz']:
			return self._get_cell_size(data_type, **kwargs)
		elif data_type == 'P':
			return np.sqrt(self.get('Px', dump=sdf_num, si=si)**2 + self.get('Py', dump=sdf_num, si=si)**2)
		elif data_type.lower() in ['zr', 'rayleigh']:
			return self._get_variable(['Z_r', 'Z_r_1'], **kwargs)
		elif data_type.lower() in ['rs', 'spot_size']:
			return self._get_variable(['W_0', 'W_0_1'], **kwargs)

		# check for directional inputs
		name = self._directional.get(data_type, data_type)
		direction = -1
		if name != data_type:
			direction = self._get_directional(data_type)
		# time needs Header to exist
		elif data_type == "time":
			name = "Header"
		# if not a directional or time
		else:
			# check to see if an alternate name was given
			name = self._alt_names.get(data_type, data_type)		
		
		# if data_type doesn't exit, _data_type_exists() will exit the program
		self._data_type_exists(name, True)

		# Header does not have an attribute data
		if data_type == "Header":
			return self._current_dict[name]

		# getting time requires a special function
		if data_type == "time":
			data = self._current_dict['Header']['time']
			# reset name equal to time for self._normalize
			name = "time"
		else:
			data = self._current_dict[name].data

		# if the input is directional return the given direction
		if direction != -1:
			try:
				data = data[direction]
			except:
				errmsg = '"%s" not defined in %s' % (data_type, self.code_name)
				raise IndexError(errmsg)

		# return a numpy array if data is an array
		if not isinstance(data, np.ndarray):
			data = np.array(data)

		# return a dimless value if SI is given as false
		if not si:
			norm = self._normalize.get(name, 1)
			# if data.copy() is not called, then any attempt to modify data results in:
			# ValueError: output array is read-only
			data = data.copy()
			data *= norm

		return data

	# read data from a range of sdf files
	# this can accept a single string, a list of strings, or any combination of the two
	# if a single data_type is given, then a single np.array will be returned
	# if multiple data_types are given then a dictionary of np.arrays will be returned
	def get_range(self, data_type, *args, **kwargs):
		begin = kwargs.get('begin', self.first)
		end = kwargs.get('end', self.last)
		hide = kwargs.get('hide', False)
		si = kwargs.get('si', self.si)

		# add +1 to end so that it is inclusive
		end += 1

		# allow data_types to be created a list or a string
		if isinstance(data_type, list):
			data_types = data_type
		else:
			data_types = [data_type]
		for arg in args:
			data_types.append(arg)

		# set up data structure
		data = {}
		for data_type in data_types:
			data[data_type] = []

		# read data
		for t in range(begin, end):
			tools.output_progress('reading data', t, end)
			for data_type in data_types:
				data[data_type].append(np.array(self.get(data_type, dump=t, si=si)))

		# ensure data contains np.arrays
		for key in data:
			data[key] = np.array(data[key])

		# do not return entire data structure if there is only 1 data type
		if len(data_types) == 1:
			return data[data_types[0]]
		else:
			return data

	# read data from all sdf files
	def get_all(self, data_type, *args, **kwargs):
		return self.get_range(data_type, *args, begin=self.first, end=self.last, **kwargs)

	# displays the value of a data_type from a sdf file
	def display(self, data_type, **kwargs):
		data = self.get(data_type, **kwargs)
		# print data in the easiest to read format
		if isinstance(data, np.ndarray):
			print '{}: {}'.format(data_type, data)
			return
		else:
			if data > 1e4 or data < 1e-4:
				print '%s: %0.12e' % (data_type, data)
				return
		print '%s: %0.12f' % (data_type, data)

	def _show_alt_names(self):
		print 'Acceptable alternate names for data types:'
		for alt_name in sorted(self._alt_names):
			print '%s = %s' % (alt_name, self._alt_names[alt_name])

	# displays all data types as well as alternate names
	def show_data_types(self):
		for data_type in self.sdf_data_types:
			print data_type
		print '\n----------------------------------------------'
		self._show_alt_names()
		if self._reference:
			print '\n----------------------------------------------'
			print 'Reference Values:'
			for k in sorted(self._reference.keys()):
				print k 


if __name__ == '__main__':
	import gc
	import get_args

	parser = get_args.GetArgs()
	path = parser.get('dir', help='Path to data')
	full = parser.get('full', False, help='Check all data types for memory leaks')

	timer = tools.Timer()
	dr = SDF_Reader(path=path)

	# do a full test of the sdf files for memory leaks
	if full:
		for data_type in dr.sdf_data_types + ['x', 'y', 'pos_x', 'pos_y']:
			mem = []
			# read the data for the first dump
			try: d = dr.get(data_type, dump=0)
			except: pass
			mem_prev = timer.get_mem()
			for t in range(50):
				#pos_x = sdf.read('/mnt/c/Users/tkawa/EPOCH/epoch2d/data/0101.sdf', dict=True)['Grid/Particles/electron'].data[0]
				try: d = dr.get(data_type, dump=t)
				except: pass

				gc.collect()
				
				mem.append(timer.get_mem() - mem_prev)
				mem_prev = timer.get_mem()
			try: print '{}: {}'.format(data_type, np.average(np.array(mem)))
			except: pass
	# 'pos_x' had the largest memory leak. Check only 'pos_x'
	else:
		# read the data for the first dump
		try: d = dr.get('pos_x', dump=0)
		except: pass
		mem_prev = timer.get_mem()
		for t in range(50):
			d = dr.get('pos_x', dump=t)

			gc.collect()
			
			print '{:^25}  diff: {}'.format(timer.memory(display=False), timer.get_mem() - mem_prev)
			mem_prev = timer.get_mem()
