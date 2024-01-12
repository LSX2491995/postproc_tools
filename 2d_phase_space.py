import numpy as np
from scipy import constants as sp
import utils.postproc_tools as tools
import utils.get_args as get_args
import spot_size
import make_movie
import utils.hilbert_curve as hilbert_curve
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.colors as mcolors
import os
import sys

def run(start, args, **kwargs):
	parser = get_args.GetArgs(args=args, start=start, overrides=kwargs, help_exit=False)
	path = parser.get('dir')
	run = parser.get('run')
	as_import = parser.get('as_import', False)
	name = parser.get('name', 'test')
	data_types = parser.get('data', ['P','Bz','Ex','Py','N','Window'], type=list)
	extent_x = parser.get('extent_x')	
	extent_y = parser.get('extent_y')	
	extent_si = parser.get('extent_si', False, help='If True, then EXTENT_X and EXTENT_Y are given in SI')
	extent_k0 = parser.get('extent_k0', False, help='If True, then EXTENT_X and EXTENT_Y are given in terms of the laser wavenumber k0')
	extent_all = parser.get('extent_all', False, help='If True, use the given extents for all plots')
	extent_laser = parser.get('extent_laser', type=tuple, len=2, help='The distance about the center of the laser to plot the x-extents. EXTENT_SI AND EXTENT_K0 still apply')
	open_movie = parser.get('open', True)
	code = parser.get('code', 'EPOCH')
	vmax = parser.get('vmax')
	time_range = parser.get('time')
	dump_range = parser.get('dump')
	cutoff = parser.get('cutoff', 20, help='All momenta values greater than <cutoff> will be plotted with a separate colorscale')
	cutoff_split = parser.get('split', help='Similar to <cutoff>, but instead of plotting each cutoff on the same plot, split them up into separate plots.')
	if cutoff_split: log = True
	else: log = False
	log = parser.get('log', log, help='Plot phase space on a log scale. Default True if <cutoff> or <split> is given')
	domain_decomp = parser.get('domain', False, help="Plot the MPI domain decomposition and Smilei patch domain. If 'Both' is given, then plot the momentum with and without the domain_decomp. Default <False>.")
	si = parser.get('si', False, help='Use S.I. units.')
	alpha = parser.get('alpha', 0.2, help='Alpha parameter for scatter plot')
	size = parser.get('size', 0.0008, alt_name='s', help='The size parameter for the scatterplot')
	bsize = parser.get('bsize', size, help='The size of the particles below a cutoff')
	balpha = parser.get('balpha', 0.015, help='Alpha parameter for the particles below a cutoff.')
	angle_lim = parser.get('angle', [], help=('Only display angles within <angle1,angle2>. '
		'If a single arg is given display <angle,angle>. Default is -45,45'))
	filter_p = parser.get('filter', help='Filter out particles with P-total < <filter>')
	t_cut = parser.get('t_cut', (-10,10), type=tuple, len=2, help='Only count particles from within this transverse cut for the title')
	max_parts = parser.get('max_p', type=int, help='If there are more particles, than MAX_P, randomly select particles so only MAX_P are plotted.')
	perc_parts = parser.get('perc_p', type=float, help='Only plot a percentage of particles equal to PERC_P, Overrides MAX_P')
	w = parser.get('w')
	parser.finish(ignore=True)
	not_parsed = parser.get_unparsed_args()
	filter_R = parser.get('filter_R', 5.8e-5)
	filter_px = parser.get('filter_px',0)
	filter_x = parser.get('filter_x',5e-6)
	filter_y = parser.get('filter_y',1.3e-4)
	filter_pp = parser.get('filter_pp',0.75)
	fa = parser.get('fa',0.5)
	
	dr = tools.get_data_reader(code=code, path=path, run=run, si=si, w=w, time=time_range, dump=dump_range,
															extent_y=extent_y, extent_x=extent_x, extent_si=extent_si, extent_k0=extent_k0)

	if domain_decomp and code == 'Smilei':
		hilbert_curve.make_decomposition(dr.path, dr.npatch_x, dr.npatch_y)

	# choose colormaps
	if cutoff or cutoff_split:
		#cmaps = ['winter', 'spring']
		cmaps = ['Blues']
	else:
		cmaps = ['Blues']

	# ensure there is the proper data to handle extent_laser
	if extent_laser is None:
		laser_extents = False
	else:
		laser_extents = True
		data_path = os.path.join(tools.get_output_path(run), os.path.dirname(name), 'laser_data.npy')
		if not os.path.exists(data_path):
			spot_size.run(0, ['plot=false', 'name={}'.format(name)])
		if not os.path.exists(data_path):
			print "could not find '{}' and 'spot_size.run()' failed".format(data_path)
			sys.exit()
		laser_data = np.load(data_path)
		# convert extents if necessary
		if extent_si is True:
			extent_laser = (extent_laser[0] * dr.kp, extent_laser[1] * dr.kp)
		if extent_k0 is True:
			extent_laser = (extent_laser[0] / dr.k_kp, extent_laser[1] / dr.k_kp)

	# handle what momentum will be plotted
	p_names = {
		'P' : 'P',
		'Px': 'Px',
		'Py': 'Py',
	}
	# create a new list to determine the order in which data will be plotted for make_movie
	order = data_types[:]
	# handle all momentum separate from other data types
	p_list = []
	for data_type in data_types[:]:
		if data_type in ['P', 'Px', 'Py']:
			p_list.append(data_types.pop(data_types.index(data_type)))
			i = order.index(data_type)
			order.insert(i, p_names[order.pop(i)])
	if len(p_list) == 0:
		p_list.append('P')

	# ensure angle_lim is a list of 2 values
	if not isinstance(angle_lim, list):
		try:
			angle_lim = [-1. * angle_lim, angle_lim]
		except:
			raise TypeError("angle='{}' must be an int or float".format(angle_lim))

	def reduce_particles(x, y, p):
		"""Given a ndarray of particles, reduce them according to MAX_P and PERC_P
		IF PERC_P is given, it takes precedence over MAX_P"""
		if p.size == 0: return x, y, p		
		if perc_parts:
			i = np.arange(p.size)
			np.random.shuffle(i)
			i = i[:int(p.size * perc_parts)]
			return x[i], y[i], p[i]
		if max_parts:
			if p.size > max_parts:
				i = np.arange(p.size)
				np.random.shuffle(i)
				i = i[:max_parts]
				return x[i], y[i], p[i]
		return x, y, p
	
	# define a function to read momentum and return the necessary data
	def get_momentum(dr, xi, y, t, i):	
		px = dr.get('Px', dump=t)[i]
		py = dr.get('Py', dump=t)[i]
		if dr.si:
			px = px/(sp.m_e*sp.c)
			py = py/(sp.m_e*sp.c)
		p_total = np.sqrt(px**2 + py**2)
		if filter_p:
			i, = np.where(p_total > filter_p)
			xi, y = xi[i], y[i]
			px, py, p_total = px[i], py[i], p_total[i]
		d = {}
		for p in p_list:
			if p == 'P':
				d[p_names[p]] = [xi, y, p_total]
			elif p == 'Px':
				d[p_names[p]] = [xi, y, px]
			elif p == 'Py':
				d[p_names[p]] = [xi, y, py]
			if cutoff != None:
				d[p_names[p]].append(p_total)
			d[p_names[p]] = tuple(d[p_names[p]])

		# plot the angle of the particles
		if angle_lim:
			# get angle information from abs(py/px) <-> abs(p_perp/p_parallel)
			i, = np.where(px > 0)
			angle = np.divide(py[i], px[i])
			angle = np.arctan(angle)
			angle = np.multiply(angle, 180. / sp.pi)

			# filter angle
			i, = np.where((angle >= angle_lim[0]) & (angle <= angle_lim[1]))
			d['angle'] = (xi[i], y[i], angle[i])

		# add 2 more data sets for each momentum type. First is all momentum
		# up to a given cutoff, second is all momentum greater than cutoff
		if cutoff_split:
			for k, v in d.items():
				_, _, p = v
				i1 = np.where(p > cutoff_split)
				i2 = np.where(p <= cutoff_split)
				d['{} > {}'.format(k, cutoff_split)] = (xi[i1], y[i1], p[i1])
				d['{} < {}'.format(k, cutoff_split)] = (xi[i2], y[i2], p[i2])

		if domain_decomp == 'Both' and code == 'Smilei':
			for k, v in d.items():
				d['{}-Domain'.format(k)] = v
		return d

	def data_func(dr, t):
		xi = dr.get('pos_xi', dump=t)

		if laser_extents is True:
			time = dr.get('time', dump=t)
			try:
				i = np.where(laser_data[0, :] == time)[0][0]
			except IndexError:
				print 'could not find laser data for time: {}'.format(time)
				ix = dr.get_ix(xi)
			else:
				laser_center = laser_data[1, i]
				ex = (laser_center + extent_laser[0], laser_center + extent_laser[1])
				ix, = np.where((xi > ex[0]) & (xi < ex[1]))
		else:
			ix = dr.get_ix(xi)

		y = dr.get('pos_y', dump=t)
		iy = dr.get_iy(y)
		i = np.intersect1d(ix, iy)
		xi = xi[i]
		y = y[i]		
				
		# create data array
		data = get_momentum(dr, xi, y, t, i)
		# add additional data_types
		for data_type in data_types:			
			if data_type.lower() in ['window', 'profile']: continue
			data[data_type] = dr.get(data_type, dump=t)
		return data

	def update_func(dr, f, im, cb, ax, ax_info, data_type):
		x, y, p = f
		if p.size == 0:
			return (im, cb, ax, ax_info)
		x, y, p = reduce_particles(x, y, p)

		# plot particles
		sc = ax.scatter(x, y, c=p, cmap='Blues', s=size, alpha=alpha)
		# handle log scales
		pmin, pmax = p.min(), p.max()
		# never log scale data < cutoff, it may have 0-valued data
		if '<' in data_type: pass
		# ensure there is a positive value and no 0 values for log plot
		elif log is True and pmax > 0 and pmin > 0:
			sc.set_norm(mcolors.LogNorm())
		
		if cb is None:
			cb = plt.colorbar(sc, ax=ax, aspect=20, shrink=0.8, pad=0.02)
		if vmax:
			cb.set_clim(vmin=pmin, vmax=vmax)
		else:
			cb.set_clim(vmin=pmin, vmax=pmax)
		cb.set_alpha(1)
		cb.draw_all()

		if dr.si:
			ax.set_xlim(left=np.min(x), right=np.max(x))
			ax.set_ylim(bottom=np.min(y), top=np.max(y))			
			#ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
			#ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

			xticks = ax.get_xticks()*1e6
			yticks = ax.get_yticks()*1e6
			ax.set_xticklabels(xticks.astype(int))
			ax.set_yticklabels(yticks.astype(int))

			ax.set_xlabel(r'$x(\mu\,m)$', labelpad=1, fontsize=9)
			ax.set_ylabel(r'$y(\mu\,m)$', labelpad=-1, fontsize=9)
		else:

			if extent_x:
			    ax.set_xlim(left=extent_x[0], right=extent_x[1])
			if extent_y:
			    ax.set_ylim(bottom=extent_y[0], top=extent_y[1])


	

		if code == 'Smilei':
			if domain_decomp == True or 'Domain' in data_type:
				ylim = ax.get_ylim()
				ax = hilbert_curve.plot_decomposition(dr, ax)
				ax.set_ylim(bottom=ylim[0], top=ylim[1])
		return (im, cb, ax, ax_info)

	def update_cutoff(dr, f, im, cb, ax, ax_info, data_type):
		x, y, p,  p_total = f
		if p.size == 0:
			return (im, cb, ax, ax_info)
			
		# add a second scatter plot to the same axis for all particles with p > cutoff

		fcoff=cutoff
		if dr.si is True:
			fcoff=0.1*np.max(p_total)

		i1, = np.where(p_total <= fcoff)			
		i2, = np.where(p_total > fcoff)

		if dr.si is True:
			fkp=1
		else:
			fkp=dr.kp

		#filter_r = filter_R*dr.kp
		#rr = np.sqrt((x-(2.5e-4)*fa*fkp)**2+y**2)
		pp = np.sqrt((p_total**2-p**2)/(p**2+0.00001*fcoff**2))
		#ii1, = np.where((p > filter_px) & (rr > filter_r) & (x > (1.25e-4*dr.kp)))
		ii1, = np.where((p > filter_px) & (pp < filter_pp) & (x > ((2.5e-4)*fa+filter_x)*fkp))
		ii2, = np.where((p > filter_px) & (pp < filter_pp) &(x > ((2.5e-4)*fa+filter_x+1e-5)*fkp))
		ii3, = np.where((p > filter_px) & (x > ((2.5e-4)*fa+filter_x)*fkp) )
		ii4, = np.where((p > filter_px) & (x > ((2.5e-4)*fa+filter_x+1e-5)*fkp) )




		#i3, = np.where(x > filter_x)
		x2, y2, p2 = x[i2], y[i2], p[i2]
		x,  y,  p  = x[i1], y[i1], p[i1]
		x, y, p = reduce_particles(x, y, p)

		

		# plot low-energy particles
		sc = ax.scatter(x, y, c='black', s=bsize, alpha=balpha)

		# if there are high-energy particles, plot them
		if p2.size > 0:			
			sc = ax.scatter(x2, y2, c=p2, cmap='Blues', s=size, alpha=alpha)
			if cb is None:
				cb = plt.colorbar(sc, ax=ax, aspect=20, shrink=0.8, pad=0.02)

			# handle log scales
			pmin, pmax = p2.min(), p2.max()
			if log and pmax > 0:
				sc.set_norm(mcolors.LogNorm())
				cb.set_norm(mcolors.LogNorm())
			else:
				sc.set_norm(mcolors.Normalize())
				cb.set_norm(mcolors.Normalize())

			if 'Py' in data_type:
				vmax = max([pmin, pmax])
				cb.set_clim(vmin=-1*vmax, vmax=vmax)
				sc.set_cmap('seismic')
			else:
				cb.set_clim(vmin=pmin, vmax=pmax)
			cb.set_alpha(1)
			cb.draw_all()		

			# find the total number of injected electrons
			if t_cut:
				count = np.where((y2 >= t_cut[0]) & (y2 <= t_cut[1]))[0]
			else:
				#count = i2
				count1 = ii1
				count2 = ii2
				count3 = ii3
				count4 = ii4
			# sum the weights of each macroparticle to get the total number of real electrons
			#injected = np.sum(dr.get('weight')[count])
			filtered1 = np.sum(dr.get('weight')[count1])
			filtered2 = np.sum(dr.get('weight')[count2])
			filtered3 = np.sum(dr.get('weight')[count3])
			filtered4 = np.sum(dr.get('weight')[count4])

			if 'Px' in data_type:
				##ax.set_title(r'$%s$ $(injected: %0.3e)$'%(data_type, injected), loc='left')
				ax.set_title(r'$%s$ $(f1: %0.3e)(f2: %0.3e)(f3: %0.3e)(f4: %0.3e)$'%(data_type, filtered1,filtered2,filtered3,filtered4), loc='left')

		if dr.si:
			ax.set_xlim(left=np.min(x), right=np.max(x))
			ax.set_ylim(bottom=np.min(y), top=np.max(y))
			#ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
			#ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
			
			xticks = ax.get_xticks()*1e6
			yticks = ax.get_yticks()*1e6
			ax.set_xticklabels(xticks.astype(int))
			ax.set_yticklabels(yticks.astype(int))

			ax.set_xlabel(r'$x(\mu\,m)$', labelpad=1, fontsize=9)
			ax.set_ylabel(r'$y(\mu\,m)$', labelpad=-1, fontsize=9)
			
		else:
			if extent_x:
			    ax.set_xlim(left=extent_x[0], right=extent_x[1])
			if extent_y:
			    ax.set_ylim(bottom=extent_y[0], top=extent_y[1])

			
		if code == 'Smilei':
			if domain_decomp == True or 'Domain' in data_type:
				ylim = ax.get_ylim()
				ax = hilbert_curve.plot_decomposition(dr, ax)
				ax.set_ylim(bottom=ylim[0], top=ylim[1])
		return (im, cb, ax, ax_info)

	# create update_func dictionary so make_movie.py knows how to handle
	# each data_type when plotting data
	update_funcs = {}
	for p in p_list:		
		if cutoff_split:
			update_funcs['{} > {}'.format(p_names[p], cutoff_split)] = update_func
			update_funcs['{} < {}'.format(p_names[p], cutoff_split)] = update_func
			update_funcs[p_names[p]] = update_func
		elif cutoff != None:
			update_funcs[p_names[p]] = update_cutoff
		elif domain_decomp == 'Both':
			update_funcs['{}-Domain'.format(p_names[p])] = update_func
		elif angle_lim:
			update_funcs['angle'] = update_func
		else:
			update_funcs[p_names[p]] = update_func

	if as_import:
		return data_func, update_func

	# if extent_all is given, pass the extents to make_movie.run()
	if extent_all is True:
		not_parsed['extent_x'] = dr.extent_x
		not_parsed['extent_y'] = dr.extent_y

	make_movie.run(args=args, dr=dr, code=code, data_func=data_func, update_func=update_funcs, data=order, extent_y=extent_y, extent_x=extent_x, si=si,
		save_as=name, path=path, open=open_movie, time=time_range, **not_parsed)

def get_update_func(**kwargs):
	"""Wrapper to get the update_func from run()."""
	_, update_func = run(None, as_import=True, **kwargs)
	return update_func

def get_data_func(**kwargs):
	"""Wrapper to get the data_func from run()."""
	data_func, _ = run(None, as_import=True, **kwargs)
	return update_func

def get_funcs(**kwargs):
	"""Wrapper to get (data_func, update_func) from run()."""
	return run(None, as_import=True, **kwargs)	
	
# if ran as a script
if (__name__ == '__main__'):
	run(1, [])
