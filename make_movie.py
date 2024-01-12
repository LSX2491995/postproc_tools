import numpy as np
import os
import sys
import subprocess
import traceback
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.animation as animation
from matplotlib import rcParams
import matplotlib.colors as mcolors
import matplotlib.cm as cmaps
import utils.get_args as get_args
import utils.postproc_tools as tools
from cycler import cycler

plt.rcParams['axes.prop_cycle'] = cycler(color='brgcmyk')
plt.rcParams['animation.ffmpeg_path'] = tools.get_ffmpeg()

def get_movie_path(movie_name, run):
	"""Get the path in which the movie will be saved to"""
	if not movie_name.endswith('.mp4'):
		movie_name = '{}.mp4'.format(movie_name)
	movie_path = os.path.join(tools.get_output_path(run), movie_name)
	output_path = os.path.dirname(movie_path)
	# if the output dir doesn't exist create it
	if os.path.isdir(output_path) is False:
		os.makedirs(output_path)	
	print 'movie path: {}'.format(movie_path)
	return movie_path

def create_movie(*args, **kwargs):
	timer = tools.Timer()
	# allow movies to be made using keyword arguments (other scripts call make_movie.py)
	# or through args (used when passing command line arguments)
	# if len(args) == 1: start = args[0]
	# else: start = len(sys.argv)
	# if len(args) == 2: args = args[1]
	try: start = args[0]
	except IndexError: start = len(sys.argv)
	try: args = args[1]
	except IndexError: args = []
	# inputs that can come from other scripts
	dr = kwargs.get('dr')
	path = kwargs.get('path')

	# parse command line arguments
	parser = get_args.GetArgs(args=args, start=start, overrides=kwargs, show_args=True)
	path        = parser.get('dir', path, help='Path to a data directory')
	run         = parser.get('run', type=str, help='the run number of a movie to make')	
	si          = parser.get('si', False, help='Use S.I. units.')
	time_range  = parser.get('time', help='A time given in units of wpt. If a single time is given, 0-TIME is used. If two times are given, use TIME1,TIME2')
	dump_range  = parser.get('dump', help='If a single dump is given, set DUMP to the final dump. If two times are given, use DUMP1,DUMP2')
	img_time    = parser.get('img', help='Create an image from a given time. This overrides TIME')
	img_dump    = parser.get('img_dump', help='Create an image from a given dump number. This overrides TIME and IMG')
	img_format  = parser.get('img_format', 'png', help='File format for saving images')
	duration    = parser.get('duration', type=float, help='Modify FPS so that a given time (in seconds) becomes the duration of the movie')
	data_types  = parser.get('data', ['Bz','Ex','Jx','N','Window','Jy'], help=('A comma separated list of data types to plot. The order given will determine the data type arrangement.'))
	save_as     = parser.get('name', "test", alt_name='save_as', type=str, help='Save the movie with NAME.')	
	open_movie  = parser.get('open', True, help='If True, open the movie after it is made.')
	skip        = parser.get('skip', 1, help='The number of dumps to advance for the next frame.')
	dim         = parser.get('dim', '2d', help='The dimensionality of the data.')
	vmax_in     = parser.get('vmax',[], help='Set the maximum value for a given data type to VMAX')
	vmin_in     = parser.get('vmin',[], help='Set the minimum value for a given data type to VMIN')
	auto_vmax   = parser.get('auto_vmax', [ ], help='Bound the upper limit to VMAX only if the data exceeds VMAX.')
	auto_vmin   = parser.get('auto_vmin', [], help='Bound the lower limit to VMIN only if the data is less than VMIN.')
	cutoff      = parser.get('cutoff', ['N',  5e-4, 'Ne',  5e-4, 'NHe', 5e-4], type=list, help='All values below the cutoff will be white. 2d data only.')
	extent_x    = parser.get('extent_x', help='Give a min and max extent in the x direction: EXTENT_X_MIN,EXTENT_X_MAX')
	extent_y    = parser.get('extent_y', help='Give a min and max extent in the Y direction: EXTENT_Y_MIN,EXTENT_Y_MAX')
	extent_k0		= parser.get('extent_k0', help='If True, then EXTENT_X and EXTENT_Y are given in terms of the laser wavenumber k0')
	extent_si 	= parser.get('extent_si', False, help='If True, then EXTENT_X and EXTENT_Y are given in SI')
	title_loc   = parser.get('title_loc', 'left', alt_name='loc', help="The location of the title on each frame: 'left', 'center', or 'right'")
	xi          = parser.get('xi', True, help='Use xi as the x-coordinate instead of x')
	log         = parser.get('log', [], help='Use a log plot for DATA_TYPE.')
	sqrt        = parser.get('sqrt', [], help='Plot the square root for DATA_TYPE')
	fps         = parser.get('fps', help='The frames per second.')
	color       = parser.get('color', 'blue', help='Plot 1d data with a given color. No effect for 2d data')
	diagnostics = parser.get('diagnostics', True, help='Display hardware usage diagnostics')
	code        = parser.get('code', 'EPOCH', help='The name of the code used to create data.')	
	w           = parser.get('w', help='wL / wp')
	parser.finish()

	# set the name for the timer diagnostics
	timer.name = save_as

	# set the appropriate data type to get and label for the x-axis
	if xi is True:
		xcoord = 'xi'
		if si: x_label = r'$\xi(m)$'
		else: x_label = r'$k_p\xi$'
	else:
		xcoord = 'x'
		if si: x_label = r'$x(m)$'
		else: x_label = r'$k_p\,x$'

	# get the data reader	
	if dr is None:
		dr = tools.get_data_reader(code=code, path=path, run=run, w=w, si=si, time=time_range, dump=dump_range,
			extent_y=extent_y, extent_x=extent_x, extent_si=extent_si, extent_k0=extent_k0, xcoord=xcoord)
	# get the path in which the movie will be saved
	movie_path = get_movie_path(save_as, run)

	# return the indices for a data array given an extent in x or y
	# these indices can be used to get fields within the given extent
	def trim(data_array, (extent1, extent2)):
			try:
				i, = np.where((data_array >= extent1) & (data_array <= extent2))    
				i1 = i[0]
				i2 = i[-1]
			except:
				# if no values satisfy extent1, use 0
				# this allows extent_x to be used alongside a moving window
				i1 = 0
				i2 = len(data_array) - 1

			# if the two indices are equal, then invalid extents were given
			if i1 == i2:
				# if xi is false, then the extents are static and misgiven
				if xi is False:
						print 'i1: {}, i2: {}'.format(i1, i2)
						print 'invalid extents given: ({},{})'.format(extent1, extent2)
						print 'extents must be within: ({},{})'.format(np.min(data_array), np.max(data_array))
						sys.exit()
				# if xi is true, then data_array will be changing with time, so do not sys.exit()
				else:
					i1 = 0
					i2 = len(data_array) - 1
			# ensure indices are returned in increasing order
			i1, i2 = sorted((i1, i2))
			return (i1, i2)

	# call trim() and set the extent indicies: ix, iy
	if extent_x:
		ix = trim(dr.get(xcoord, dump=dr.first), dr.extent_x)
		#ix = dr.get_ix(dr.get(xcoord, dump=dr.first))
	else:
		ix = (0, dr.num_cells_x)			
	if dim == '2d':
		if extent_y:
			iy = trim(dr.get('y', dump=dr.first), dr.extent_y)
			#iy = dr.get_iy(dr.get('y', dump=dr.first))
		else:
			iy = (0, dr.num_cells_y)
			extent_y = (dr.get('y', dump=dr.first)[iy[0]], dr.get('y', dump=dr.first)[iy[1]])
			
	# a list of data types to ignore because they're plots are handled elsewhere
	ignore_data_types = ['window', 'profile']
	# get a data_function from another script
	# this will be called at each timestep in order to update the data for the next frame
	data_func = kwargs.get('data_func')
	# create an data func if none is given
	if data_func is None:
		if data_types:
			# ensure data types is a list
			if not isinstance(data_types, list): data_types = [data_types]			
			def data_func(dr, t):
				update_data = {}
				for data_type in data_types:
					if data_type.lower() in ignore_data_types: continue
					update_data[data_type] = dr.get(data_type, dump=t)
				return update_data
		else:
			print 'Error: data_func and data_types not given.'
			sys.exit()
	# create data types array to track which data types will be used for the movie
	else:
		# ensure data_types reflects data
		if data_types:
			for k in data_func(dr, dr.first).keys():
				if k not in data_types:
					data_types.insert(0, k)
		else:
			data_types = []
			for data_type in sorted(data_func(dr, dr.first).keys()):
				data_types.append(data_type)	

	# determine if the moving window is going to be plotted
	if 'window' in [d.lower() for d in data_types]:
		plot_window = True
	else:
		plot_window = False

	# get the density profile over the entire domain
	if plot_window is True:
		# initialize density
		Ne = []
		# initialize x
		x_whole = []
		# initialize the time array associated with Ne
		time_Ne = []
		# find the middle index
		i_mid = dr.num_cells_y / 2
		for t in range(dr.abs_first, dr.abs_last+1):
			if not dr.has_dump(t):continue
			tools.output_progress('preparing to make movie', t, dr.abs_first, dr.abs_last+1)
			Ne.append(dr.get('Ne', dump=t)[-2][i_mid])
			x_whole.append(dr.get('x', dump=t)[-1])
			time_Ne.append(t)			
		Ne = np.array(Ne)
		x_whole = np.array(x_whole)
		time_Ne = np.array(time_Ne)

	# plot_funcs has keys which are data_types and values which are functions that modify plot()
	# this allows each script to define its own behavior for the initial plot
	plot_funcs = kwargs.get('plot_func', {})
	# update_funcs has keys which are data_types and values which are functions that modify update()
	# this allows each script to define its own behavior for updating the movie
	update_funcs = kwargs.get('update_func', {})
	# ax_func should be function with the signature: ax_func(fig, data_type) and returns
	# a matplotlib axis object attached to the given figure
	ax_func = kwargs.get('ax_func')

	# build the extent data structure
	# currently static extents for all data types. This can be updated
	# so that each data type can be given its own unique extents
	extent = {}
	for data_type in data_types:
		if dim == '1d':
			extent[data_type] = dr.extent_x
		else:
			extent[data_type] = [dr.extent_x, dr.extent_y]

	# build the vmax/vmin data structure. Each data type can have a unique vmin/vmax
	vmax = {}
	for i in range(0, len(vmax_in)):
		if isinstance(vmax_in[i], str):
			if vmax_in[i] not in vmax:
				vmax[vmax_in[i]] = vmax_in[i+1]
	vmin = {}
	for i in range(0, len(vmin_in)):
		if isinstance(vmin_in[i], str):
			if vmin_in[i] not in vmin:
				vmin[vmin_in[i]] = vmin_in[i+1]

	# set the time given by img_time. if img_time is a bool (i.e. given as a flag)
	# then the time_range of the data reader does not need to be changed
	if img_time != None and not isinstance(img_time, bool):
		# if a single time is given, plot a single frame
		if not isinstance(img_time, list): 
			img_time = (img_time, img_time)
		dr.set_time_range(img_time)
	# set the time given by img_dump which overrides img_time
	if img_dump != None and not isinstance(img_dump, bool):
		# if a single time is given, plot a single frame
		if not isinstance(img_dump, list): 
			img_dump = (img_dump, img_dump)
		dr.set_time_range(img_dump, isdump=True)
		img_time = None



	############ Functions for updating/plotting each frame #########################################################
	# movie making functions
	def update2d(data_type, f, t):
		im, cb, ax, ax_info = plots[data_type]
		# if extents are given, trim data
		f = f[ix[0]:ix[1], iy[0]:iy[1]]
		# check for an update_func		
		update_func = update_funcs.get(data_type)
		if update_func:			
			im, cb, ax, ax_info = update_func(dr, f, im, cb, ax, ax_info, data_type)
		else:			
			# default handling for fields
			if xi is True:
				im.set_array(np.fliplr(np.flipud(f.T)))
			else:
				im.set_array(np.flipud(f.T))
			# update longitudinal axis
			ex = extent[data_type]
			temp_x = dr.get(xcoord, dump=t)
			ex[0] = sorted([temp_x[ix[0]], temp_x[ix[1]]])
			im.set_extent((ex[0][0], ex[0][1], ex[1][0], ex[1][1]))

			# set title			
			if data_type in sqrt:
				ax.set_title(r'$%s-sqrt$' % data_type, loc=title_loc)
			else:
				ax.set_title(r'$%s$' % data_type, loc=title_loc)

			# check vmax for auto-scaling 
			if data_type in auto_vmax:
				i = auto_vmax.index(data_type)
				try:
					auto_scale_vmax = auto_vmax[i+1]
				except:
					raise IndexError("auto_scale for '{}' not properly given".format(data_type))

			# check vmin for auto-scaling
			if data_type in auto_vmin:
				i = auto_vmin.index(data_type)
				try:
					auto_scale_vmin = auto_vmin[i+1]
				except:
					raise IndexError("auto_scale for '{}' not properly given".format(data_type))

			# set colorbar maximum
			if data_type in vmax:
				vmax_temp = vmax[data_type]
			else:
				vmax_temp = f.max()
				if data_type in sqrt: vmax_temp = np.sqrt(vmax_temp)
				# specifying vmax will override auto_vmax. Only check for auto_vmax if vmax is not given.
				if data_type in auto_vmax:
					if vmax_temp > auto_scale_vmax:	
						ax.set_title(r'$%s$ $(Max:%0.3f)$' % (data_type, vmax_temp), loc=title_loc)					
						vmax_temp = auto_scale_vmax						
			# set colorbar minimum
			if data_type in vmin:
				vmin_temp = vmin[data_type]
			elif data_type in cutoff:
				i = cutoff.index(data_type)
				try:
					vmin_temp = cutoff[i+1]
				except:
					raise IndexError("cutoff for '{}' not properly given".format(data_type))
			else:				
				vmin_temp = f.min()
				if data_type in sqrt: vmin_temp = np.sqrt(vmin_temp)					
				# specifying vmin will override auto_vmin. Only check for auto_vmin if vmin is not given.
				if data_type in auto_vmin:
					if vmin_temp < auto_scale_vmin:
						vmin_temp = auto_scale_vmin

			cb.set_clim(vmin=vmin_temp, vmax=vmax_temp) 
			cb.draw_all()
		return im, cb, ax, ax_info
	
	def update1d(data_type, f, t):		
		im, cb, ax, ax_info = plots[data_type]

		# a script can return 'NOCLEAR' to define this behavior
		if ax_info != 'NOCLEAR': ax.clear()

		# axis labels
		# do this before plotting in case update_func overrides this info
		ax.set_xlabel(x_label, labelpad=1, fontsize=9)
		ax.set_title(r'$%s$' % data_type, loc=title_loc)
		ax.xaxis.tick_bottom()
		ax.yaxis.tick_left()

		# check for an update_func
		update_func = update_funcs.get(data_type)		
		if update_func:
			im, cb, ax, ax_info = update_func(dr, f, im, cb, ax, ax_info, data_type)
			plots[data_type] = (im, cb, ax, ax_info)
		# default behavior for updating the movie
		else:
			x, y = f
			x, y = x[ix[0]:ix[1]], y[ix[0]:ix[1]]
			ax.plot(x, y, c=color, lw=0.5)
		return im, cb, ax, ax_info	

	def update(data_func, t):
		data = data_func(dr, t)
		for data_type in data_types:
			if data_type.lower() in ignore_data_types: continue
			# 1d data if f is a tuple
			f = data[data_type]
			if isinstance(f, tuple) or dim == '1d':
				plots[data_type] = update1d(data_type, f, t)
			else:
				plots[data_type] = update2d(data_type, f, t)

	def plot2d(data_type, f, ax):
		# check for an update_func		
		update_func = update_funcs.get(data_type)
		if update_func:
			im, cb, ax, ax_info = update_func(dr, f, None, None, ax, ax_info, data_type)
		else:
			f = f[ix[0]:ix[1], iy[0]:iy[1]]
			ax_info = None
			# if extent_x is given, use that
			if extent_x:
				ex = extent[data_type]
			
			# if extent_x is not given, find it
			else:
				ex = extent[data_type]
				temp_x = dr.get(xcoord, dump=dr.first)
				ex[0] = [np.min(temp_x), np.max(temp_x)]

			if data_type in sqrt:
				f = np.sqrt(f)

			if data_type in log:
				norm = mcolors.LogNorm()
			else:
				norm = None			

			# check for cutoff
			if data_type in cutoff:
				i = cutoff.index(data_type)
				try:
					vmin = cutoff[i+1]
				except:
					raise IndexError("cutoff for '{}' not properly given".format(data_type))
			else:
				vmin = None

			# create colormap
			cmap = cmaps.gist_rainbow
			cmap.set_under('white')
			im = plt.imshow(np.fliplr(np.flipud(f.T)), extent=(ex[0][0], ex[0][1], ex[1][0], ex[1][1]), 
					cmap=cmap, vmin=vmin, aspect="auto", interpolation='none', norm=norm)

			cb = plt.colorbar(im, ax=ax, aspect=20, shrink=0.8, pad=0.02)
		#plt.gca().set_autoscale_on(True)
		return im, cb, ax, ax_info

	def plot1d(data_type, f, ax):
		return None, None, ax, None

	def plot(data_type, ax):
		f = data_func(dr, dr.first)[data_type]
		# check for an plot_func		
		plot_func = plot_funcs.get(data_type)
		if plot_func:
			return plot_func(f, None, None, ax, None, data_type)

		# 1d data if f is a tuple
		if isinstance(f, tuple) or dim == '1d':
			return plot1d(data_type, f, ax)
		else:
			return plot2d(data_type, f, ax)	

	def window(t):
		"""Function to update the moving window position"""
		x = dr.get('x', dump=t)
		i = np.where(time_Ne == t)[0][0]
		na = Ne[i:i+1].sum()
		x1 = x[0]
		x2 = x[-1]
		return [x2 - ((x2 - x1) / 2.), x2], [na, na]



	############ Setup the figure/subplots ##############################################################################
	# set up all subplots
	num_data_types = len(data_types)
	plots = {}

	# set up the number of rows and columns for subplots
	if num_data_types == 4:
		r = 2
		c = 2
	else:
		r = 1
		c = 1
		max_cols = 3
		while (r * c < num_data_types):
			c += 1
			if c > max_cols:
				c = max_cols - 1
				r += 1

	# if there is more than 1 row, create more space for the plots
	if num_data_types == 1:
		fig_size = (8., 6.)
	elif r == 1:
		fig_size = (15., 12.5 / c)
	elif r == 2 and c == 2:
		fig_size = (15., 6.5)
	else:
		fig_size = (15., 6.)
	# create the figure
	fig = plt.figure(figsize=fig_size)
	fig.subplots_adjust(left=0.08, right=0.96, top=0.85, bottom=0.1, wspace=0.2, hspace=0.3)

	# set up all plots
	for i, data_type in enumerate(data_types):
		if ax_func:
			ax = ax_func(fig, data_type)
		else:
			ax = plt.subplot(r, c, i+1)
		# plot the density profile from a data set with a moving window
		if data_type.lower() == 'window':			
			ax.plot(x_whole[:-1], Ne[:-1], lw=0.5)
			if dr.si:
				ax.set_ylabel(r'$y(m)$', labelpad=-10, fontsize=9)
				plt.xlabel(r'$x(m)$', labelpad=1, fontsize=9)
			else:
				plt.xlabel(r'$k_p\,x$', labelpad=1, fontsize=9)
			ax.set_title(r'$N_i$', loc=title_loc)
			wz, wn = window(dr.first)
			l, = ax.plot(wz, wn, 'r')
		# plot the density profile from a data set with without a moving window
		elif data_type.lower() == 'profile':
			# plot the lineout at y=0
			#N = dr.get('N', dump=0)
			N = dr.get('Ne', dump=0)
			iy_0 = np.where(dr.get('y', dump=0) >= 0)[0][0]
			ax.plot(dr.get('x')[1:], N[:,iy_0], lw=0.5)
			if dr.si:
				plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
				plt.xlabel(r'$x(m)$', labelpad=1, fontsize=9)
			else:
				plt.xlabel(r'$k_p\,x$', labelpad=1, fontsize=9)
			ax.set_title(r'$N_i$', loc=title_loc)
		# handle the remaining data types
		else:
			ax.xaxis.tick_bottom()
			ax.yaxis.tick_left()
			ax.set_title(r'$%s$' % data_type, loc=title_loc)
			ax.set_xlabel(x_label, labelpad=1, fontsize=9)
			if dim == '1d' and 'phase space' in data_type:
				ax.set_ylabel(r'$p_x (MeV/c)$', labelpad=-10, fontsize=9)
			else:
				if dr.si:
					plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
					plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
					ax.set_ylabel(r'$y(m)$', labelpad=-10, fontsize=9)
					ax.set_xlabel(r'$x(m)$', labelpad=-10, fontsize=9)
				else:
					ax.set_ylabel(r'$k_p\,y$', labelpad=-10, fontsize=9)
			plots[data_type] = plot(data_type, ax)
			



	############ Create an image from a frame(s) #################################################################
	# create images at particular points in time
	if img_time or img_dump:
		for t in range(dr.first, dr.last+1, skip):
			if dr.has_dump(t) is False: continue
			if extent_x:
				ix = trim(dr.get(xcoord, dump=t), dr.extent_x)
			update(data_func, t)

			if plot_window is True:
				wz, wn = window(t)
				l.set_xdata(wz)
				l.set_ydata(wn)

			if dr.si is True:
				fig.suptitle(r'$t = {:.6e}$'.format(dr.get('time', dump=t, si=True)), y=0.94)
			else:
				fig.suptitle(r'$\omega_p\,t = {}$'.format(dr.get('time', dump=t)), y=0.94)

			label = t
			if img_time:
				label = dr.get('time', dump=t)
			img_path = '{0}-{1:4>0}'.format(movie_path.replace('.mp4', ''), label)
			# fig.savefig(img_path, dpi=300)
			# print 'image saved: {}'.format(img_path)
			img_path = tools.save_image(fig, img_path, format=img_format)
			if open_movie is True:
				tools.open_file(img_path)
			timer.stop()
		sys.exit()



	############ Make the movie #########################################################################################
	# create movie writer
	FFMpegWriter = animation.writers['ffmpeg']

	# set the fps
	if duration:
		num_frames = dr.last - dr.first + 1
		fps = num_frames / duration
	elif fps is None:
		fps = 5
		num_dumps = dr.last - dr.first + 1
		# alter framerate based on number of dumps using 150 dumps as the reference
		if num_dumps > 150:
			fps *= ((num_dumps / skip) * 0.5) / 150.
	# ensure fps is greater than 0
	if fps <= 0: fps = 5

	# set metadata
	if dr.run_data:
		metadata = {'title': '{} ({})'.format(os.path.basename(movie_path), dr.run_data[0])}
	else:
		metadata = {'title': os.path.basename(movie_path)}
	#metadata = dict(title='Double Jet a0 = %(a0)g, L = %(L)g'%params[uid], artist='uid: %s'%d.uid(), comment=d.uid())
	writer = FFMpegWriter(fps=fps, metadata=metadata, bitrate=-1)

	# include the final dump
	end = dr.last + 1
	with writer.saving(fig, movie_path, 300):
		for t in range(dr.first, end, skip):
			if not dr.has_dump(t): continue
			tools.output_progress('making movie', t, dr.first, end, skip=skip)
			if extent_x:
				ix = trim(dr.get(xcoord, dump=t), dr.extent_x)
			update(data_func, t)

			if plot_window is True:
				wz, wn = window(t)
				l.set_xdata(wz)
				l.set_ydata(wn)
			
			if dr.si is True:
				plt.suptitle(r'$t = {:.6e}$'.format(dr.get('time', dump=t, si=True)), y=0.94)
			else:
				plt.suptitle(r'$\omega_p\,t = {}$'.format(dr.get('time', dump=t)), y=0.94)			

			# check plot info to see if any plots returned IGNORE
			if 'IGNORE' not in [p[3] for p in plots.values()]:				
				writer.grab_frame()

			# take a snapshot of the resource usage for the process
			if diagnostics is True:
				try: timer.snapshot(t)
				except: 
					diagnostics = False
					print 'Unable to take a shapshot of process resource usage'
	plt.close(fig)



	############ Handle final diagnostics #######################################################################################3333
	# display information
	if open_movie is True: tools.open_file(movie_path)
	if diagnostics is True: 
		try: timer.display()
		except: print 'Unable to display process resource usage'
	else: timer.stop()

def run(*args, **kwargs):
	try:
		create_movie(*args, **kwargs)
	except Exception as e:
		print 'Failed to make movie'
		traceback.print_exc()

# if ran as a script
if __name__ == "__main__":
	run(1, [])



############ Code to add text to the figure ###############
# info is a list of tuples that contains parameters for the given data
# it will be added to the movie as text
# info = dr.get_input('a0', 'bc_y_max', 'buffer_y', bc_y_max='bc_y', silent=True)
# info.insert(0, ('dx', dr.get('dx')))
# if dr.dim > 1):
# 	info.insert(1, ('dy', dr.get('dy')))
# info.insert(2, ('n0', dr.n0))

# # if info is given, then add the text to the figure
# x_text = 0.01
# x_inc = 0.2
# y_text = 0.15
# y_inc = 0.035
# for i in info:
# 	text = '%s = %s' % (i[0], i[1])
# 	plt.text(x_text, y_text, text, fontsize=7, transform=plt.gcf().transFigure)
# 	y_text -= y_inc
# 	# if the y coordinate is less than 0 (won't show up on the figure), then reset to original y val and increment x
# 	if y_text < 0):
# 		y_text = 0.15
# 		x_text += x_inc