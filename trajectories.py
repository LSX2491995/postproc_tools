import numpy as np
from scipy import constants as sp
import utils.postproc_tools as tools
import utils.get_args as get_args
import make_movie
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import os
import sys

def run(start, args, **kwargs):
	timer = tools.Timer()
	parser = get_args.GetArgs(args=args, start=start, overrides=kwargs)
	path = parser.get('dir')
	run = parser.get('run', type=str, help='The run number for which data will be used')
	name = parser.get('name', "test")
	data_types = parser.get('data', ['Window'], type=list, help="Data types in addition to P, Py, and the trajectory plot. Default 'Window'")
	extent_x = parser.get('extent_x', (530, 535), help='The x range to determine what particles to track')	
	extent_y = parser.get('extent_y', (-10,10), help='The y range to determine what particles to track')
	extent_x_plot = parser.get('extent_x_plot', (520, 550), help='The x range to determine what particles to plot for referencing chosen particles.')	
	extent_y_plot = parser.get('extent_y_plot', (-15,15), help='The y range to determine what particles to plot for referencing chosen particles.')
	extent_x_traj = parser.get('extent_x_traj', help='The x range to set the axis limits of the trajectory plot.')	
	extent_y_traj = parser.get('extent_y_traj', (-10,10), help='The y range to set the axis limits of the trajectory plot.')
	time = parser.get('track_time', 1750, help='The TIME used to determine what particles to track')
	filter_p = parser.get('filter', 20, help='Do not track particles with P-total < <filter>')
	perc = parser.get('perc', 1., help='Float between 0-1 that determines what percentage of filtered particles to track. Default 0.1')
	pos = parser.get('pos', False, help='Only track particles with p-perp > 0')
	neg = parser.get('neg', False, help='Only track particles with p-perp < 0')
	time_range = parser.get('time', help='The time in which to plot trajectories over')
	alpha = parser.get('alpha', 0.3, help='Alpha parameter for scatter plot. Default 0.3')
	size = parser.get('size', 0.05, help='The size parameter for the scatterplot. Default 0.05')
	talpha = parser.get('talpha', 0.05, help='Alpha parameter for trajectory plot. Default 0.05')
	linewidth = parser.get('lw', 0.5, help='The linewidth of the trajectory plot. Default 0.5')
	scolor = parser.get('scolor', 'lime', help="The color to plot the tracked particles. Default 'lime'")
	code = parser.get('code', 'EPOCH')
	si = parser.get('si', False)
	skip = parser.get('skip', 1, help='The number of dumps from the previous for the next trajectory point. Default 1')
	movie = parser.get('movie', False, help='If true, make a movie that tracks the selected particles.')
	save_avgs = parser.get('save_avgs', True, help='If True, save the averages data')
	plot_avgs = parser.get('plot_avgs', help='If True, plot the saved data in DIR')
	plot_time = parser.get('plot_time', False, help="If True, plot the trajectories vs time")
	include_selected = parser.get('include_selected', True, help='Include a plot of the selected particles on the trajectories plot. Default True.')
	num_bins = parser.get('bins', help='An integer value of the number of bins to split extent_x into')
	fit = parser.get('fit', help='If true, fit trajectory to a sin wave. Can give 2 x-positions to fit a subset')
	save_trajectories = parser.get('save', False, help='If true, save the selected particles trajectories')
	parser.finish(ignore=True)
	not_parsed = parser.get_unparsed_args()

	############### Plot the Average Trajectories ###############
	if plot_avgs:
		path = os.path.join(tools.get_output_path(run), path)
		if name == 'test':
			name = os.path.basename(path)
		# try: run, data = plot_avgs.split('-', 1)
		# except: path = os.path.join(path, plot_avgs)
		# else: path = os.path.join(path, '{}-series/{}/trajectories'.format(run, plot_avgs))
		
		fig, ax = plt.subplots(dpi=300, figsize=(15,6))
		ax.set_title('Mean Trajectories')
		ax.set_xlabel(r'$k_p\,x$')
		ax.set_ylabel(r'$k_p\,y$')
		data_path = os.path.join(path, 'data')
		for file in os.listdir(data_path):
			print file
			if file.endswith('.npy'):
				d = np.load(os.path.join(data_path, file))
				ax.plot(d[0], d[1], label=file[:-4])
		ax.legend(bbox_to_anchor=(1.125, 1))
		ax.set_ylim(bottom=-7, top=7)
		save_path = os.path.join(path, 'mean_trajectories_{}.png'.format(name))
		plt.savefig(save_path, dpi=300)
		print 'mean trajectories saved: {}'.format(save_path)
		# exit trajectories
		sys.exit()
	############ Create Multiple Processes for Bins #############
	if num_bins:
		# only round integers 
		if (extent_x[1]-extent_x[0])%num_bins == 0:
			bins = np.linspace(extent_x[0], extent_x[1], num_bins+1, dtype=int)
		else:
			bins = np.linspace(extent_x[0], extent_x[1], num_bins+1)

		# create the commands to start a process to handle an individual bin
		commands = []
		for i in range(0, bins.size-1):
			b1, b2 = bins[i], bins[i+1]
			n = '{}/{}-{}'.format(name, b1, b2)
			script_path = os.path.join(tools.get_script_path(), 'trajectories.py')
			args = ['python', script_path, 'extent_x={},{}'.format(b1, b2), 'name={}/{}-{}'.format(name, b1, b2)]
			for arg, val in parser.get_args().items():
				if arg in ['bins', 'extent_x', 'name']: continue
				args.append('{}={}'.format(arg,val))
			commands.append(args)
			
		# make dir
		data_path = os.path.join(tools.get_output_path(run), name)
		if not os.path.exists(data_path):
			os.makedirs(data_path)
			print 'made dir: {}'.format(data_path)		
		# # submit the commands to the queue
		tools.to_queue(*commands)
		sys.exit()
	#############################################################

	# begin the creation of the trajectories
	if perc < 0 or perc > 1: 
		raise ValueError('perc must be a float between 0-1, not: {}'.format(perc))

	if pos and neg:
		raise ValueError('both pos and neg cannot be used')

	dr = tools.get_data_reader(code=code, path=path, run=run, si=si, w=parser.get('w'), time=time_range,
		extent_y=extent_y, extent_x=extent_x)

	# assign min/max values for the fit
	if fit is True:
		fit = [dr.xmin, dr.xmax]

	# make the trajectories dir
	full_name = name
	data_path = os.path.join(tools.get_output_path(run), name)
	name = os.path.basename(data_path)
	data_path = os.path.dirname(data_path)
	if not os.path.exists(data_path):
		os.makedirs(data_path)
		print 'made dir: {}'.format(data_path)
	
	# select particles of interest
	def select_ids():
		print 'Selecting ids...'
		select_timer = tools.Timer()
		dump = dr.get_dump(time)
		ids = dr.get('id', dump=dump)
		xi = dr.get('pos_xi', dump=dump)
		y = dr.get('pos_y', dump=dump)	
		ix, = np.where((xi > extent_x_plot[0]) & (xi < extent_x_plot[1]))
		iy, = np.where((y > extent_y_plot[0]) & (y < extent_y_plot[1]))
		i = np.intersect1d(ix, iy)	

		# filter particles by position
		xi, y, ids = xi[i], y[i], ids[i]
		
		# filter particles by momenta
		px, py = dr.get('Px', dump=dump)[i], dr.get('Py', dump=dump)[i]
		p = np.sqrt(px**2 + py**2)
		i2, = np.where(p > filter_p)
		
		# set i as the indices of all current particles
		i = np.linspace(0, xi.size-1, xi.size, dtype=int)
		i = np.delete(i, i2)

		# ensure there are particles to track
		if i2.size == 0:
			print 'paramters and/or data resulted in no particles to track'			
			sys.exit()
		# show the user what particles are being tracked
		# determine the min and max. Use the higher of 2 for the vmin/vmax
		vmax = max([py[i2].min(), py[i2].max()])		
		f, axarr = plt.subplots(2, sharex=True, figsize=(16, 12), dpi=300)	
		if include_selected is True:
			f2, axarr2 = plt.subplots(2, figsize=(16,12), dpi=300)
			axarr = np.append(axarr, axarr2[1])	
		for j, ax in enumerate(axarr):
			if j == 0:
				ax.set_title(r'Tracked Particles at $\omega_p\,t={}$'.format(time))
			ax.set_xlabel(r'$k_p\xi$')
			ax.set_ylabel(r'$k_p\,y$')
			# i contains indices for particles below filter_p
			ax.scatter(xi[i], y[i], c='black', alpha=0.3, s=0.005)
			# i2 contains indices for particles above filter_p
			sc = ax.scatter(xi[i2], y[i2], c=py[i2], cmap='seismic', s=0.1, alpha=0.5)
			cb = plt.colorbar(sc, ax=ax, aspect=20, shrink=0.8, pad=0.02)
			cb.set_clim(vmin=vmax*-1, vmax=vmax)
			cb.set_alpha(1)
			cb.draw_all()

		# filter particles by momentum
		i = i2
		xi, y, ids, py = xi[i], y[i], ids[i], py[i]		

		# filter particles by position for selection
		ix = dr.get_ix(xi)
		iy = dr.get_iy(y)
		i = np.intersect1d(ix, iy)
		xi, y, ids, py = xi[i], y[i], ids[i], py[i]

		# filter out positive or negative particles		
		if pos:
			i, = np.where(py > 0)
			xi, y, ids, py = xi[i], y[i], ids[i], py[i]
		elif neg:
			i, = np.where(py < 0)
			xi, y, ids, py = xi[i], y[i], ids[i], py[i]

		# set i as the indices of all current particles
		i = np.linspace(0, xi.size-1, xi.size, dtype=int)

		# randomly select n ids	
		n = int(i.size * perc)
		i = np.random.choice(i, size=n, replace=False)
		xi, y, ids, py = xi[i], y[i], ids[i], py[i]

		# plot the new particles overtop the old
		sc2 = axarr[0].scatter(xi, y, c=scolor, s=0.1, alpha=1)

		# save plot
		save_path = os.path.join(data_path, 'selected/{}_selected.png'.format(name))
		# make output dir
		try: os.makedirs(os.path.dirname(save_path))
		# dir already exists	
		except OSError: pass
		else: print 'made dir: {}'.format(os.path.dirname(save_path))		
		f.savefig(save_path, dpi=300)
		print 'plot saved at: {}'.format(save_path)
		select_timer.stop('finished selection')
		if include_selected:
			axarr2[1].set_title(r'Tracked Particles at $\omega_p\,t={}$'.format(time))
			axarr2[1].scatter(xi, y, c=scolor, s=0.1, alpha=1)
			return ids, f2, axarr2[0]
		else:
			return ids, None, None
	
	# begin tracking particles
	selected_ids, fig, ax = select_ids()
	print '# of particle ids: {}'.format(selected_ids.size)	

	# determine the name for the trajectories
	if pos is False and neg is False:
		trajectory_title = r'Particle Trajectories'
		py_str = 'all'
	elif pos is True:
		trajectory_title = r'Particle Trajectories (Py > 0)'
		py_str = 'Py > 0'
	elif neg is True:
		trajectory_title = r'Particle Trajectories (Py < 0)'
		py_str = 'Py < 0'
	# include the y range
	trajectory_title = r'{} (y-extent={},{})'.format(trajectory_title, extent_y[0], extent_y[1])

	# list of data types to determine the order in which the data types are displayed
	order = [trajectory_title, 'P', 'Py']
	order.extend(data_types)

	def data_func(dr, t):
		data = {}
		# get the window position for the trajectory plot
		xmin = dr.get('x', dump=t)[0]
		data[trajectory_title] = (dr.get('x', dump=t)[0])

		# get particle data
		ids = dr.get('id', dump=t)
		xi = dr.get('pos_xi', dump=t)
		y = dr.get('pos_y', dump=t)	
		ix, = np.where((xi > extent_x_plot[0]) & (xi < extent_x_plot[1]))
		iy, = np.where((y > extent_y_plot[0]) & (y < extent_y_plot[1]))
		i = np.intersect1d(ix, iy)

		# filter particles by position
		xi, y, ids = xi[i], y[i], ids[i]		
		# filter particles by momenta
		px, py = dr.get('Px', dump=t)[i], dr.get('Py', dump=t)[i]
		p = np.sqrt(px**2 + py**2)

		# i contains indices for particles above filter_p
		i, = np.where(p > filter_p)		
		# i2 contains indices for particles below filter_p
		i2 = np.delete(np.linspace(0, xi.size-1, xi.size, dtype=int), i)
		# i3 contains the selected particles
		i3 = np.isin(ids, selected_ids).nonzero()[0]

		data['P'] = (i, i2, i3, xi, y, p)
		data['Py'] = (i, i2, i3, xi, y, py)
		
		for data_type in data_types:
			if data_type.lower() == 'window': continue
			data[data_type] = dr.get(data_type, dump=t)
		return data

	def plot_func(f, im, cb, ax, ax_info, data_type, no_movie=False):
		# if plot_func is called with no axes, create it
		if ax is None:
			f, ax = plt.subplots(dpi=300, figsize=(15,6))
			new_fig = True
		if no_movie is True:
			new_fig = True

		# read particle data across all timesteps
		x = []
		y = []
		p = []
		# track the min/max particle energy
		vmin = None
		vmax = None
		# keep the position average
		x_avg = []
		y_avg = []
		for t in range(dr.first, dr.last+1, skip):
			tools.output_progress('making trajectories', t, dr.first, dr.last+1, skip=skip)
			i = np.isin(dr.get('id', dump=t), selected_ids).nonzero()[0]
			if i.size == 0: continue
			# calculate particle energy and determine min/max
			p_tmp = np.sqrt(dr.get('Px')[i]**2 + dr.get('Py')[i]**2)
			p_min, p_max = p_tmp.min(), p_tmp.max()
			if vmin is None or p_min < vmin: vmin = p_min
			if vmax is None or p_max > vmax: vmax = p_max
			# get particle position and calculate the average
			x_tmp, y_tmp = dr.get('pos_x')[i], dr.get('pos_y')[i]
			x.extend(x_tmp)
			y.extend(y_tmp)
			p.extend(p_tmp)
			x_avg.append(np.average(x_tmp))
			y_avg.append(np.average(y_tmp))
		# plot data
		x = np.array(x)
		y = np.array(y)
		p = np.array(p)
		sc = ax.scatter(x, y, c=p, cmap='plasma', alpha=alpha, s=size)

		# plot the averages
		ax.plot(x_avg, y_avg, c='lime', lw=0.8)
		# save the averages if using bins
		if save_avgs:
			save_path = os.path.join(data_path, 'data/{}.npy'.format(name))
			# make output dir
			try: os.makedirs(os.path.dirname(save_path))
			# dir already exists	
			except OSError: pass
			else: print 'made dir: {}'.format(os.path.dirname(save_path))
			# save averages
			try: np.save(save_path, np.array([x_avg, y_avg]))
			except IOError as e: raise e
			else: print 'averages saved: {}'.format(save_path)

		# make colorbar
		cb = plt.colorbar(sc, ax=ax, aspect=20, shrink=0.8, pad=0.02)
		cb.set_clim(vmin=vmin, vmax=vmax)
		cb.set_alpha(1)
		cb.draw_all()

		# save plot
		ax.set_title(r'{} (# particles={})'.format(trajectory_title, selected_ids.size), loc='left')
		ax.set_xlabel(r'$k_p\,x$')
		ax.set_ylabel(r'$k_p\,y$')
		ax.set_ylim(bottom=extent_y_traj[0], top=extent_y_traj[1])
		save_path = os.path.join(data_path, '{}.png'.format(name))

		if no_movie is True:
			f.savefig(save_path, dpi=300)
		else:
			# Get the portion of the figure that this plot is in
			extent = ax.get_window_extent().transformed(plt.gcf().dpi_scale_trans.inverted())
			extent = extent.expanded(1.1, 1.5)
			points = extent.get_points()
			points[0][0] = 0.01
			points[1][0] = 8.
			extent.set_points(points)
			plt.savefig(save_path, bbox_inches=extent, dpi=300)
		print 'plot saved at: {}'.format(save_path)

		if fit:
			i = np.where((x >= fit[0]) & (x <= fit[1]))
			x, y, p = x[i], y[i], p[i]
			save_path = os.path.join(data_path, 'data/{}-full.npy'.format(name))
			# save particle data
			try: np.save(save_path, np.array([x, y, p]))
			except IOError:
				try: os.makedirs(os.path.dirname(save_path))
				except OSError as e: print '{}\nparticle data was not saved'.format(repr(e))
				else:
					print 'made dir: {}'.format(os.path.dirname(save_path))
					np.save(save_path, np.array([x, y, p]))
					print 'particle data saved: {}'.format(save_path)
			else: print 'particle data saved: {}'.format(save_path)

		# add the vertical line for the moving window after saving the trajectories
		try: line = ax.axvline(f, lw=0.5, c='red')
		except: line = None
		return (line, cb, ax, 'NOCLEAR')

	def update_trajectories(f, line, cb, ax, ax_info, data_type):
		line.set_xdata(f)
		return (line, cb, ax, ax_info)

	def update_func(dr, f, im, cb, ax, ax_info, data_type):
		# i has the high energy particles
		# i2 has the low energy particles
		# i3 has the particles selected for tracking
		i, i2, i3, xi, y, p = f
		
		ax.set_xlabel(r'$k_p\xi$')
		ax.set_ylabel(r'$k_p\,y$')
		# plot the low energy particles
		ax.scatter(xi[i2], y[i2], c='black', alpha=0.1, s=0.005)

		# ensure there are high energy particles
		if i.size > 0:
			# plot the high energy particles
			if data_type == 'Py':
				# determine the min and max. Use the higher of 2 for the vmin/vmax
				vmax = max([p[i].min(), p[i].max()])
				vmin = -1 * vmax
				sc = ax.scatter(xi[i], y[i], c=p[i], cmap='seismic', s=size, alpha=alpha)
				# plot the selected particles
				ax.scatter(xi[i3], y[i3], c=scolor, alpha=1, s=size)
			else:
				vmin, vmax = p[i].min(), p[i].max()
				sc = ax.scatter(xi[i], y[i], c=p[i], cmap='plasma', s=size, alpha=alpha)
				# plot the selected particles
				ax.scatter(xi[i3], y[i3], c=scolor, alpha=1, s=size)

			# create the colorbar for the first time
			if cb is None:
				cb = plt.colorbar(sc, ax=ax, aspect=20, shrink=0.8, pad=0.02)
			cb.set_clim(vmin=vmin, vmax=vmax)
			cb.draw_all()		
		return (im, cb, ax, ax_info)

	def trajectories_time():
		# if plot_func is called with no axes, create it		
		fig, axarr = plt.subplots(2, figsize=(16, 12), dpi=300)

		# track the min/max particle energy
		vmin = None
		vmax = None
		# keep the position average
		x_avg = []
		y_avg = []
		t_avg = []
		for t in range(dr.first, dr.last+1, skip):
			tools.output_progress('making trajectories', t, dr.first, dr.last+1, skip=skip)
			i = np.isin(dr.get('id', dump=t), selected_ids).nonzero()[0]
			if i.size == 0: continue
			# calculate particle energy and determine min/max
			p = np.sqrt(dr.get('Px')[i]**2 + dr.get('Py')[i]**2)
			p_min, p_max = p.min(), p.max()
			if vmin is None or p_min < vmin: vmin = p_min
			if vmax is None or p_max > vmax: vmax = p_max
			# get particle position and calculate the average
			x, y = dr.get('pos_x')[i], dr.get('pos_y')[i]
			x_avg.append(np.average(x))
			y_avg.append(np.average(y))
			t_avg.append(dr.get('time', dump=t))
			t_arr = np.ones(x.size) * dr.get('time', dump=t)
			axarr[0].scatter(x, y, c=p, cmap='plasma', alpha=alpha, s=size)
			axarr[1].scatter(t_arr, y, c=p, cmap='plasma', alpha=alpha, s=size)

		# plot the averages
		axarr[0].plot(x_avg, y_avg, c='lime', lw=0.8)
		axarr[1].plot(t_avg, y_avg, c='lime', lw=0.8)
		# save the averages if using bins
		if save_avgs:
			save_path = os.path.join(data_path, 'data/{}.npy'.format(name))
			try: np.save(save_path, np.array([x_avg, y_avg, t_avg]))
			except IOError:
				try: os.makedirs(os.path.dirname(save_path))
				except OSError as e: print '{}\naverages were not saved'.format(repr(e))
				else:
					print 'made dir: {}'.format(os.path.dirname(save_path))
					np.save(save_path, np.array([x_avg, y_avg, t_avg]))
					print 'averages saved: {}'.format(save_path)
			else: print 'averages saved: {}'.format(save_path)

		# iterate through each PathCollection obj and set the normalization for colorbar
		norm = plt.Normalize(vmin, vmax)
		for ax in axarr:
			for sc in ax.collections:
				#sc.set_clim(vmin=vmin, vmax=vmax)
				sc.set_norm(norm)
		cb = plt.colorbar(sc, ax=axarr[0], aspect=20, shrink=0.8, pad=0.02)
		cb2 = plt.colorbar(sc, ax=axarr[1], aspect=20, shrink=0.8, pad=0.02)
		cb.set_clim(vmin=vmin, vmax=vmax)
		cb2.set_clim(vmin=vmin, vmax=vmax)
		cb.set_alpha(1)
		cb2.set_alpha(1)
		cb.draw_all()		
		cb2.draw_all()

		# save plot
		axarr[0].set_title(r'{} (# particles={})'.format(trajectory_title, selected_ids.size), loc='left')
		axarr[0].set_xlabel(r'$k_p\,x$')
		axarr[0].set_ylabel(r'$k_p\,y$')
		axarr[0].set_ylim(bottom=extent_y_traj[0], top=extent_y_traj[1])
		axarr[1].set_xlabel(r'$\omega_p\,t$')
		axarr[1].set_ylabel(r'$k_p\,y$')
		axarr[1].set_ylim(bottom=extent_y_traj[0], top=extent_y_traj[1])
		#axarr[1].set_ylim(bottom=dr.extent_y[0], top=dr.extent_y[1])
		save_path = os.path.join(data_path, '{}.png'.format(name))

		fig.savefig(save_path, dpi=300)
		print 'plot saved at: {}'.format(save_path)


	def save_trajectory_data():
		# read particle data across all timesteps
		x = []
		y = []
		px = []
		py = []
		p = []
		# keep the position average
		x_avg = []
		y_avg = []
		for t in range(dr.first, dr.last+1, skip):
			tools.output_progress('making trajectories', t, dr.first, dr.last+1, skip=skip)
			i = np.isin(dr.get('id', dump=t), selected_ids).nonzero()[0]
			if i.size == 0: continue			
			# get particle data
			x_tmp, y_tmp = dr.get('pos_x')[i], dr.get('pos_y')[i]
			px_tmp = dr.get('Px')[i]
			py_tmp = dr.get('Py')[i]
			p_tmp = np.sqrt(px_tmp**2 + py_tmp**2)
			# add data to arrays
			x.extend(x_tmp)
			y.extend(y_tmp)
			px.extend(px_tmp)
			py.extend(py_tmp)
			p.extend(p_tmp)
			x_avg.append(np.average(x_tmp))
			y_avg.append(np.average(y_tmp))

		# turn all arrays into numpy arrays
		x = np.asarray(x)
		y = np.asarray(y)
		px = np.asarray(px)
		py = np.asarray(py)
		p = np.asarray(p)
		x_avg = np.asarray(x_avg)
		y_avg = np.asarray(y_avg)

		# save data
		save_path = os.path.join(data_path, 'data/{}.npy'.format(name))
		save_avgs_path = os.path.join(data_path, 'data/{}_avgs.npy'.format(name))
		# make output dir
		try: os.makedirs(os.path.dirname(save_path))
		# dir already exists	
		except OSError: pass
		else: print 'made dir: {}'.format(os.path.dirname(save_path))
		# save trajectory data
		try: np.save(save_path, np.array([x, y, px, py, p, extent_y, py_str, selected_ids.size, ' '.join(sys.argv)]))
		except IOError as e: raise e
		else: print 'trajectory data saved: {}'.format(save_path)
		# save averages
		try: np.save(save_avgs_path, np.array([x_avg, y_avg, extent_y, py_str, selected_ids.size, ' '.join(sys.argv)]))
		except IOError as e: raise e
		else: print 'averages saved: {}'.format(save_avgs_path)
	################ End Functions ################

	
	# do not make a movie if plot_time is given
	if movie:
		plot_funcs = {trajectory_title: plot_func}
		update_funcs = {
			trajectory_title: update_trajectories,
			'P': update_func,
			'Py': update_func
		}
		make_movie.run(dr=dr, code=code, data_func=data_func, plot_func=plot_funcs, update_func=update_funcs, 
			data=order, save_as=full_name, path=path, time=time_range, skip=skip, **not_parsed)
	else:
		timer = tools.Timer()
		if save_trajectories is True:
			save_trajectory_data()		
		elif plot_time is True:
			trajectories_time()		
		elif include_selected is True:
			plot_func(fig, None, None, ax, None, None, no_movie=True)
		else:
			plot_func(None, None, None, None, None, None, no_movie=True)
		timer.stop('plotting trajectories')

# if ran as a script
if (__name__ == '__main__'):
	run(1, [])