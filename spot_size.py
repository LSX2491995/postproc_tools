import numpy as np
import utils.postproc_tools as tools
import utils.get_args as get_args
from matplotlib import pyplot as plt
plt.switch_backend('agg')
import os


def run(start, args, **kwargs):
	parser = get_args.GetArgs(args=args, start=start, overrides=kwargs, show_args=True)
	path = parser.get('dir')
	run = parser.get('run', type=str)
	dump = parser.get('dump')
	save_as = parser.get('name', 'spot_size', alt_name='save_as')
	field = parser.get('field', 'Bz', help='The field in which the laser intensity should be inferred from.')
	plot = parser.get('plot', True, help='If True, make a plot, otherwise only save laser data')
	open_plot = parser.get('open', True, help='Attempt to open the plot once created.')

	# get the name of the code being run and its associated reader
	code = parser.get('code', 'EPOCH')
	w = parser.get('w')
	dr = tools.get_data_reader(code=code, path=path, w=w)
	
	def calc(t):
		# get data
		x = dr.get('xi', dump=t)
		y = dr.get('y', dump=t)
		# Ex = dr.get('Ex', dump=t)
		# Ey = dr.get('Ey', dump=t)
		# E = np.sqrt(Ex**2 + Ey**2)
		# E2 = E**2
		Bz = dr.get(field, dump=t)		
		B2 = Bz**2

		sum_x = 0
		sum_x2 = 0
		sum_y = 0
		sum_y2 = 0
		sum_0 = 0
		for i in range(0, x.size-1):
			for j in range(0, B2[0,:].size):
				#fields = E2[i,j] + B2[i,j]
				fields = B2[i,j]
				sum_x += fields * x[i]
				sum_x2 += fields * x[i]**2
				sum_y += fields * y[j]
				sum_y2 += fields * y[j]**2
				sum_0 += fields
		
		# expected values
		x_exp = sum_x / sum_0
		x2_exp = sum_x2 / sum_0
		y_exp = sum_y / sum_0
		y2_exp = sum_y2 / sum_0

		# variance = <x^2> - <x>^2
		x_var = x2_exp - x_exp**2
		y_var = y2_exp - y_exp**2
		return x_exp, x_var, y_exp, y_var

	# if a single dump is given, then display information for that dump
	if dump:
		x_exp, x_var, y_exp, y_var = calc(dump)
		print '<x>: {}'.format(x_exp)
		print 'Var(x): {}'.format(x_var)
		print '<y>: {}'.format(y_exp)
		print 'Var(y): {}'.format(y_var)

	# if no dump is given, then create a plot of Var(y) vs time
	else:
		timer = tools.Timer()		
		time = []
		x_exp = []
		x_var = []
		y_exp = []
		y_var = []
		for t in range(dr.first, dr.last):
			tools.output_progress('calculating', t, dr.first, dr.last)			
			vals = calc(t)
			check_nan = np.isnan(vals)
			if True in check_nan: continue
			ex, vx, ey, vy = vals
			time.append(dr.get('time', dump=t))
			x_exp.append(ex)
			x_var.append(vx)
			y_exp.append(ey)
			y_var.append(vy)			

		time = np.array(time)
		x_exp = np.array(x_exp)
		x_var = np.array(x_var)
		y_exp = np.array(y_exp)
		y_var = np.array(y_var)		

		# save laser data
		data_path = os.path.join(tools.get_output_path(run), os.path.dirname(save_as), 'laser_data.npy')
		# make output dir
		try: os.makedirs(os.path.dirname(data_path))
		# dir already exists	
		except OSError: pass
		# save laser data
		try: np.save(data_path, np.array([time, x_exp, x_var, y_exp, y_var]))
		except IOError as e: raise e
		else: print 'laser data saved: {}'.format(data_path)

		if plot is True:
			# plot y-variance
			plt.subplots(figsize=(20,6), dpi=300)
			plt.subplot(121)
			plt.plot(time, y_var)
			plt.title(r'Spot Size'.format(time))
			plt.xlabel(r'$\omega_p\,t$')
			plt.ylabel(r'y-variance')

			# plot x expected value
			plt.subplot(122)
			plt.plot(time, x_exp)
			plt.title(r'Laser x Expected Value vs Time'.format(time))
			plt.xlabel(r'$\omega_p\,t$')
			plt.ylabel(r'Laser x Expected Value')

			# # plot y-variance
			# plt.subplots(figsize=(20,6))
			# plt.subplot(131)
			# plt.plot(time, y_var)
			# plt.title(r'Spot Size ($\omega_p\,t$)'.format(time))
			# plt.xlabel(r'$\omega_p\,t$')
			# plt.ylabel(r'$y-variance$')

			# # plot y-variance
			# plt.subplot(132)
			# plt.plot(x_exp, y_var)
			# plt.title(r'Spot Size ($k_p\,x$)'.format(time))
			# plt.xlabel(r'$k_p\,x$')
			# plt.ylabel(r'$y-variance$')			
			# 
			# # fit r^2 = r0^2(1 + (x - x0)^2 / Zr^2) to a parabola
			# b = y_var
			# a = np.ones((len(x_exp), 4))
			# for i, x in enumerate(x_exp):
			# 	#a[i][0] = 2.
			# 	a[i][1] = x**2
			# 	a[i][2] = -2. * x

			# x = np.linalg.lstsq(a, b)[0]

			# print 'x: {}'.format(x)
			# # this is the spot size of the intensity, convert to spot size of the fields
			# # r0_fields = r0_intensity / sqrt(2) 
			# r02 = x[0]# * 2.
			# r0 = np.sqrt(r02)
			# alpha = x[1]
			# #Zr2 = (2. * r02) / alpha
			# Zr2 = r02 / alpha
			# Zr = np.sqrt(Zr2)
			# x0 = x[2] / alpha
			# r2 = r02 * (1 + (x_exp - x0)**2 / Zr2)

			# print 'r02: {}'.format(r02)
			# print 'alpha: {}'.format(alpha)
			# print 'Zr2: {}'.format(Zr2)
			# print 'r0: {}'.format(r0)
			# print 'Zr: {}'.format(Zr)
			# print 'x0: {}'.format(x0)
			# print 'r2: {}'.format(r2)


			# plt.subplot(133)
			# #plt.plot(x_exp, y_var)
			# plt.plot(x_exp, r2)
			# title = 'r0 = {:.2f} Zr = {:.2f} x0 = {:.2f}'.format(r0, Zr, x0)
			# plt.title(r'Spot Size Fit ({})'.format(title))
			# plt.xlabel(r'$k_p\,x$')
			# plt.ylabel(r'$y-variance$')
			# plt.legend()

			if save_as:
				save_path = os.path.join(tools.get_output_path(run), '{}.png'.format(save_as))
				plt.savefig(save_path)
				print 'plot saved at: {}'.format(save_path)
				if open_plot is True:
					tools.open_file(save_path)
			else:
				plt.show()
		timer.stop()




if __name__ == '__main__':
	run(1, [])