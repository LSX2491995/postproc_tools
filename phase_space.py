import numpy as np 
from scipy import constants as sp
import utils.postproc_tools as tools
import utils.get_args as get_args
import make_movie
import sys

def run(start, args, **kwargs):
	parser = get_args.GetArgs(args=args, start=start, overrides=kwargs)
	path = parser.get('dir', help='Path to a data directory')
	run = parser.get('run', type=str)
	data_types = parser.get('data', ['Bz','Ey','Ex', 'Window'], type=list, help=('A comma separated list of data types to plot. '
			'The order given will determine the data type arrangement. Default Bz,Ey,Ex,Window'))
	name = parser.get('name', "test", help='Save the movie with NAME. Default "test".')
	filter_p = parser.get('filter_p', 1.971638, help='Filter out all particles below given momenta')
	plot_xi = parser.get('xi', True, help='Use xi as the x-coordinate instead of x. Default TRUE.')
	all_momentum = parser.get('all', False, help='If True, include all particles regardless of momenta')
	open_movie = parser.get('open', True, help='Attempt to open the movie once created.')
	transverse = parser.get('y', 0, help='The y position to create the line out plots')
	t_cut = parser.get('t_cut', (-10, 10), type=tuple, len=2, help='Only plot phase space particles from within this transverse cut')
	num_bins = parser.get('bins', 75, help='The number of bins for the Electron Spectrum ')	
	log = parser.get('log', True, help='Electron Spectrum counts will be in a log scale')
	# data reader args
	code = parser.get('code', 'EPOCH')	
	si = parser.get('si', False, help='Use S.I. units.')
	alpha = parser.get('alpha', 0.25, help='Alpha parameter for scatter plot')
	size = parser.get('size', 1, alt_name='s', help='The size parameter for the scatterplot')
	w = parser.get('w')
	extent_x = parser.get('extent_x')	
	extent_y = parser.get('extent_y')	
	time_range = parser.get('time')
	dump_range = parser.get('dump')
	parser.finish(ignore=True)
	not_parsed = parser.get_unparsed_args()

	dr = tools.get_data_reader(code=code, path=path, run=run, si=si, w=w, time=time_range, 
															dump=dump_range, extent_y=extent_y, extent_x=extent_x)

	# determine if the electron spectrum should be made
	if 'Spectrum' in data_types:
		make_spectrum = True
	else:
		make_spectrum = False

	# scan the data and find the min/max values for the electron spectrum
	if make_spectrum:
		def get_min_max():
			p_min = None
			p_max = None
			for t in range(dr.first, dr.last):
				tools.output_progress('fixing axes', t, dr.first, dr.last)
				p = dr.get('P', dump=t)
				try:
					temp = np.min(p)
					if p_min is None or temp < p_min:
						p_min = temp
				except: pass
				try:
					temp = np.max(p)
					if p_max is None or temp > p_max:
						p_max = temp
				except: pass
			return p_min, p_max
		p_min, p_max = get_min_max()
		# get the ranges for the electron spectrum histogram
		ranges = np.linspace(p_min, p_max, num_bins+1)

	def data_func(dr, t):
		data = {}		
		# read the total momentum
		p = dr.get('P', dump=t)

		# create the electron spectrum data
		if make_spectrum:
			spectrum = []
			for i in range(0, len(ranges)-1):
				i, = np.where((p >= ranges[i]) & (p < ranges[i+1]))
				spectrum.append(p[i])
			data['Spectrum'] = spectrum

		# read the x-position for the particles
		if plot_xi is True:
			pos_x = dr.get('pos_xi', dump=t)
		else:
			pos_x = dr.get('pos_x', dump=t)

		# get all particles that aren't at rest
		if all_momentum is True:
			i, = np.where(np.abs(p) > 0)			
		# get a list of indeces that have a momentum greater than filter_p
		else:	
			i, = np.where(p > filter_p)
		
		# only keep the desired data
		p = p[i]
		pos_x = pos_x[i]

		# get a transverse cut of particles
		if t_cut:
			y = dr.get('pos_y', dump=t)[i]
			i, = np.where((y >= t_cut[0]) & (y <= t_cut[1]))
			# only keep the desired data
			p = p[i]
			pos_x = pos_x[i]		

		# fill data
		if code == 'EPOCH':
			xi = dr.get('xi', dump=t)[1:]
		else:
			xi = dr.get('xi', dump=t)


		# get y-index for y=0
		iy = np.where(dr.get('y') >= transverse)[0][0]
		data['Phase Space'] = (pos_x, p)
		for data_type in data_types:
			if data_type in ['Window', 'Profile', 'Spectrum']: continue
			data[data_type] = (xi, dr.get(data_type, dump=t)[:,iy])
		return data

	def update_func(dr, f, im, cb, ax, ax_info, data_type):
		x, y = f
		ax.scatter(x, y, c='b', s=size, alpha=alpha)
		return (im, cb, ax, ax_info)

	def update_spectrum(dr, f, im, cb, ax, ax_info, data_type):
		n, bins, patches = ax.hist(f, bins=num_bins, histtype='stepfilled', range=(p_min, p_max), log=log, color=['b']*num_bins)
		ax.set_xlabel('gamma')
		return (im, cb, ax, ax_info)

	# create update_func_dict
	update_funcs = {'Phase Space': update_func,
									'Spectrum': update_spectrum
								 }

	# data: a dictionary containing pre-loaded data for make_movie.py
	# names: if names is not given, then the names will be taken from the data dictionary keys
	# vmax: by default vmax is set to the maximum value of the data being plotted. This is handled
	#       in the function update1d() or update2d() in make_movie.py
	# save_as: this is the name of the movie. This name will be appended with ".mp4" and added
	#          to the path of the output directory where all movies are saved
	# path: this is the path to the directory in which data will be read from
	# dim: by default the dim=2d. phase space plots require 1d
	make_movie.run(code=code, data_func=data_func, update_func=update_funcs, save_as=name,
		data=data_types[:], path=path, run=run, dim='1d', open=open_movie, **not_parsed)
	
	
# if ran as a script
if (__name__ == '__main__'):
	run(1, [])