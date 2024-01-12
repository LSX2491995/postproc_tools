import numpy as np
from scipy import constants as sp
import utils.postproc_tools as tools
import utils.get_args as get_args
import os
import sys
import matplotlib.pyplot as plt
import make_movie
plt.switch_backend('agg')
if tools.get_os() == "Linux":
	plt.switch_backend('agg')
import matplotlib.colors as mcolors



def run(start, args, **kwargs):
	timer = tools.Timer()

	parser = get_args.GetArgs(args=args, start=start, overrides=kwargs, show_args=True)
	path = parser.get('dir')
	run = parser.get('run', type=str)
	name = parser.get('name', "test")
	code = parser.get('code', 'EPOCH')
	time_range = parser.get('time')	
	si = parser.get('si', False, help='Use S.I. units.')
	alpha = parser.get('alpha', 0.3, help='Alpha parameter for scatter plot.')
	size = parser.get('size', 0.02, help='The size parameter for the scatterplot.')	
	filtered_p = parser.get('p', 10, help='Set the momenta that determines an filtered particle')
	parser.finish(ignore=True)
	not_parsed = parser.get_unparsed_args()
	filter_px = parser.get('filter_px',0)
	filter_x = parser.get('filter_x',5e-6)
	filter_pp = parser.get('filter_pp',0.75)
	fa = parser.get('fa',0.5)
	mass_fa = parser.get('mass_fa',4)

	extent_x    = parser.get('extent_x',(-1e-22, 5e-21))
	extent_y    = parser.get('extent_y')
	open_movie = parser.get('open', True)
	num_bins = parser.get('bins', 175, help='The number of bins. Default 45') # 46 bins for x_bins
	log = parser.get('log', False, help='Histogram counts will be in a log scale. Default True')
	use_stats = parser.get('stats', True, help='Create a plot of rms and mean vs time. Default True')
	stats_range = parser.get('stats_range', [530,545], help='A range within or equal to extent_x to use for stats.')
	stats_plot_ranges = parser.get('stat_ranges', [530,545,530,535,535,540,540,545], help='List of ranges within extent_x. Every two args will be coupled into a range.')
	parser.finish(ignore=True)
	
	order=['count1', 'count2', 'count3', 'count4' ]
	dr = tools.get_data_reader(code=code, path=path, run=run, si=si, w=parser.get('w'), time=time_range,extent_x=extent_x, extent_y=extent_y)

	# store particle ids of particles that have already been filtered
	filtered_ids = []
	# store the x position of filtered particles
	filtered_x = []
	# store the y position of filtered particles
	filtered_y = []
	
	
	tt = []
	filtered1 = []
	filtered2 = []
	filtered3 = []
	filtered4 = []

	filtered_ids1 = []
	filtered_ids2 = []
	filtered_ids3 = []
	filtered_ids4 = []
	#stats = {'Px1_count': {}, 'Px2_count': {}, 'Px3_count': {}, 'Px4_count': {}, 'P1_count': {}, 'P2_count': {}, 'P3_count': {}, 'P4_count': {}}



		# filter particles by momenta
	px = dr.get('HePx', dump=dr.last)
	py = dr.get('HePy', dump=dr.last)
	ids = dr.get('Heid', dump=dr.last)
	p = np.sqrt(px**2 + py**2)
	i, = np.where(p > 0)
	x = dr.get('Hepos_x', dump=dr.last)
	y = dr.get('Hepos_y', dump=dr.last)

	fkp=dr.kp

	if dr.si is True:
		fkp=1

	fcoff=filtered_p
	if dr.si is True:
		fcoff=0.1*np.max(p)

	pp = np.abs(py/(px+0.00001*fcoff))
	#ii1, = np.where((p > filter_px) & (rr > filter_r) & (x > ((2.5e-4)*fa*dr.kp)))
	i1, = np.where((px > filter_px) & (pp < filter_pp) & (x > ((2.5e-4)*fa+filter_x)*fkp))
	i2, = np.where((px > filter_px) & (pp < filter_pp) &(x > ((2.5e-4)*fa+filter_x+1e-5)*fkp))
	i3, = np.where((px > filter_px) & (x > ((2.5e-4)*fa+filter_x)*fkp) )
	i4, = np.where((px > filter_px) & (x > ((2.5e-4)*fa+filter_x+1e-5)*fkp) )


	filtered_ids1 = ids[i1]
	filtered_ids2 = ids[i2]
	filtered_ids3 = ids[i3]
	filtered_ids4 = ids[i4]

	#order = ['Px1 count', 'Px2 count', 'Px3 count', 'Px4 count', 'P1 count', 'P2 count', 'P3 count', 'P4 count']
	# iterate through all timesteps
	for t in range(dr.first, dr.last+1):
		tools.output_progress('finding filtered particles', t, dr.first, dr.last+1)

		ii1 = np.isin(dr.get('Heid', dump=t), filtered_ids1).nonzero()[0]
		ii2 = np.isin(dr.get('Heid', dump=t), filtered_ids2).nonzero()[0]
		ii3 = np.isin(dr.get('Heid', dump=t), filtered_ids3).nonzero()[0]
		ii4 = np.isin(dr.get('Heid', dump=t), filtered_ids4).nonzero()[0]





		filtered1.append(np.sum(dr.get('Heweight',dump=t)[ii1]))
		filtered2.append(np.sum(dr.get('Heweight',dump=t)[ii2]))
		filtered3.append(np.sum(dr.get('Heweight',dump=t)[ii3]))
		filtered4.append(np.sum(dr.get('Heweight',dump=t)[ii4]))
		tt.append(dr.get('time', dump=t))

		

	if dr.si is True:
		bmax = np.max((np.sqrt((px*sp.c)**2+(1836*mass_fa*sp.m_e*sp.c**2)**2)-1836*mass_fa*sp.m_e*sp.c**2)*6.424e12)
		bmin = 0
	else:
		bmax = np.max(px)
		bmin = -0.01*bmax

	# plot particles
	def data_func(dr, t):
	

		jj1 = np.isin(dr.get('Heid', dump=t), filtered_ids1).nonzero()[0]
		jj2 = np.isin(dr.get('Heid', dump=t), filtered_ids2).nonzero()[0]
		jj3 = np.isin(dr.get('Heid', dump=t), filtered_ids3).nonzero()[0]
		jj4 = np.isin(dr.get('Heid', dump=t), filtered_ids4).nonzero()[0]

		px = dr.get('HePx', dump=t)
		py = dr.get('HePy', dump=t)
		p = np.sqrt(px**2 + py**2)

		px_t1 = p[jj1]
		px_t2 = p[jj2]
		px_t3 = p[jj3]		
		px_t4 = p[jj4]
		

		mean1 = np.mean(px_t1)
		mean2 = np.mean(px_t2)
		mean3 = np.mean(px_t3)
		mean4 = np.mean(px_t4)

		if dr.si is True:

			#Calculate kenetic energy in MeV from px in kg*m/s
			px_t1 = (np.sqrt((px_t1*sp.c)**2+(1836*mass_fa*sp.m_e*sp.c**2)**2)-1836*mass_fa*sp.m_e*sp.c**2)*6.424e12
			px_t2 = (np.sqrt((px_t2*sp.c)**2+(1836*mass_fa*sp.m_e*sp.c**2)**2)-1836*mass_fa*sp.m_e*sp.c**2)*6.424e12
			px_t3 = (np.sqrt((px_t3*sp.c)**2+(1836*mass_fa*sp.m_e*sp.c**2)**2)-1836*mass_fa*sp.m_e*sp.c**2)*6.424e12
			px_t4 = (np.sqrt((px_t4*sp.c)**2+(1836*mass_fa*sp.m_e*sp.c**2)**2)-1836*mass_fa*sp.m_e*sp.c**2)*6.424e12

		# new list of weights for the filtered electrons
		ww1 = dr.get('Heweight',dump=t)[jj1]
		ww2 = dr.get('Heweight',dump=t)[jj2]
		ww3 = dr.get('Heweight',dump=t)[jj3]
		ww4 = dr.get('Heweight',dump=t)[jj4]

		

		#bins = np.linspace(dr.extent_x[0], dr.extent_x[1], num_bins+1)
		bins = np.linspace(bmin, bmax, num_bins+1)
		#Px1_count = []
		#Py1_count = []
		#P1_count = []
		#Px2_count= []
		#Py2_count = []
		#P2_count = []
		#Px3_count = []
		#Py3_count = []
		#P3_count = []
		#Px4_count = []
		#Py4_count = []
		#P4_count = []

		
		Px1_count = np.zeros(num_bins)
		Py1_count = np.zeros(num_bins)
		P1_count = np.zeros(num_bins)
		Px2_count= np.zeros(num_bins)
		Py2_count = np.zeros(num_bins)
		P2_count = np.zeros(num_bins)
		Px3_count = np.zeros(num_bins)
		Py3_count = np.zeros(num_bins)
		P3_count = np.zeros(num_bins)
		Px4_count = np.zeros(num_bins)
		Py4_count = np.zeros(num_bins)
		P4_count = np.zeros(num_bins)

		l_len1 = np.zeros(num_bins)
		l_len2 = np.zeros(num_bins)
		l_len3 = np.zeros(num_bins)
		l_len4 = np.zeros(num_bins)



		# rms = np.zeros(num_bins)
		for i in range(0, num_bins):
			ix1, = np.where((px_t1 >= bins[i]) & (px_t1 < bins[i+1]))
			#ip1, = np.where((p_t1 >= bins[i]) & (p_t1 < bins[i+1]))
			temp_px1 = px_t1[ix1]
			#temp_p1 = p_t1[ix1]
		
			ix2, = np.where((px_t2 >= bins[i]) & (px_t2 < bins[i+1]))
			#ip2, = np.where((p_t2 >= bins[i]) & (p_t2 < bins[i+1]))
			temp_px2 = px_t2[ix2]
			#temp_p2 = p_t2[ix2]

			ix3, = np.where((px_t3 >= bins[i]) & (px_t3 < bins[i+1]))
			#ip3, = np.where((p_t3 >= bins[i]) & (p_t3 < bins[i+1]))
			temp_px3 = px_t3[ix3]
			#temp_p3 = p_t3[ix3]

			ix4, = np.where((px_t4 >= bins[i]) & (px_t4 < bins[i+1]))
			#ip4, = np.where((p_t4 >= bins[i]) & (p_t4 < bins[i+1]))
			temp_px4 = px_t4[ix4]
			#temp_p4 = p_t4[ix4]

		# calculate statistics
			#Px1_count.append(np.sum(dr.get('weight',dump=t)[ix1]))
			#P1_count.append(np.sum(dr.get('weight',dump=t)[ip1]))
			#Px2_count.append(np.sum(dr.get('weight',dump=t)[ix2]))
			#P2_count.append(np.sum(dr.get('weight',dump=t)[ip2]))
			#Px3_count.append(np.sum(dr.get('weight',dump=t)[ix3]))
			#P3_count.append(np.sum(dr.get('weight',dump=t)[ip3]))
			#Px4_count.append(np.sum(dr.get('weight',dump=t)[ix4]))
			#P4_count.append(np.sum(dr.get('weight',dump=t)[ip4]))

			Px1_count[i] = np.sum(ww1[ix1])
			#P1_count[i] = np.sum(ww1[ip1])
			Px2_count[i] = np.sum(ww2[ix2])
			#P2_count[i] = np.sum(ww2[ip2])
			Px3_count[i] = np.sum(ww3[ix3])
			#P3_count[i] = np.sum(ww3[ip3])
			Px4_count[i] = np.sum(ww4[ix4])
			#P4_count[i] = np.sum(ww4[ip4])

			l_len1[i]= len(ix1)
			l_len2[i]= len(ix2)
			l_len3[i]= len(ix3)
			l_len4[i]= len(ix4)

		total_px1=np.sum(Px1_count)
		total_px2=np.sum(Px2_count)
		total_px3=np.sum(Px3_count)
		total_px4=np.sum(Px4_count)

		ll1=len(px_t1)
		ll2=len(px_t2)
		ll3=len(px_t3)
		ll4=len(px_t4)

		lll1=len(jj1)
		lll2=len(jj2)
		lll3=len(jj3)
		lll4=len(jj4)

		#llll1=np.sum(l_len1)
		#llll2=np.sum(l_len2)
		#llll3=np.sum(l_len3)
		#llll4=np.sum(l_len4)

		llll1=np.sum(dr.get('Heweight',dump=t)[jj1])
		llll2=np.sum(dr.get('Heweight',dump=t)[jj2])
		llll3=np.sum(dr.get('Heweight',dump=t)[jj3])
		llll4=np.sum(dr.get('Heweight',dump=t)[jj4])

		

			#Px1,Px2,Px3,Px4,P1,P2,P3,P4 = Px1[i],Px2[i],Px3[i],Px4[i],P1[i],P2[i],P3[i],P4[i]
			
			# rms[i] = np.sqrt(np.mean(temp_py**2))
			#if bins[i] >= extent_x[0] and bins[i] <= extent_x[1]:
			#	time = dr.get('time', dump=t)
			#	stats['Px1'].setdefault(bins[i], []).append((time, Px1[i]))
			#	stats['P1'].setdefault(bins[i], []).append((time, P1[i]))
			#	stats['Px2'].setdefault(bins[i], []).append((time, Px2[i]))
			#	stats['P2'].setdefault(bins[i], []).append((time, P2[i]))
			#	stats['Px3'].setdefault(bins[i], []).append((time, Px3[i]))
			#	stats['P3'].setdefault(bins[i], []).append((time, P3[i]))
			#	stats['Px4'].setdefault(bins[i], []).append((time, Px4[i]))
			#	stats['P4'].setdefault(bins[i], []).append((time, P4[i]))
			#	# stats['rms'].setdefault(bins[i], []).append((time, rms[i]))		

		data = {
			'count1': (bins[:-1], Px1_count, total_px1,ll1,lll1,llll1),
			#'P1 count': (bins[:-1], P1_count),
			'count2': (bins[:-1], Px2_count, total_px2, ll2,lll2,llll2),
			#'P2 count':  (bins[:-1], P2_count),
			'count3': (bins[:-1], Px3_count , total_px3, ll3,lll3,llll3),
			#'P3 count': (bins[:-1], P3_count),
			'count4': (bins[:-1], Px4_count , total_px4, ll4,lll4,llll4),
			#'P4 count': (bins[:-1], P4_count)
			#'bins plot': bins[:-1]


		}
		return data


	

	def update_func(dr, f, im, cb, ax, ax_info, data_type):		
		if data_type == 'bins plot':
			x = f
			ax.plot(x)

		else:

			pppx,count,total, llen, lllen, llllen= f
		#if x.size == 0: return (im, cb, ax, 'IGNORE')
		#count = np.where((px > filter_px) & (x > filter_x))
		#count = np.where(x > filter_x)

			
		#count = i1
		#filtered = np.sum(dr.get('weight')[count])
		#n, bins, patches = ax.hist(x, bins=num_bins, log=log)
			ax.plot(pppx, count)
			ax.set_title(r'$%s$ $(N1: %0.3e)(N2: %0.3e)$'%(data_type, total, llllen), loc='left')
			if dr.si is True:				
				ax.set_xlabel('KE(MeV)')

			else:				
				#ax.set_xlim(left=extent_x[0], right=extent_x[1])
				ax.set_xlabel('px')
		return (im, cb, ax, None)

	# create update_func dictionary so make_movie.py knows how to handle
	# each data_type when plotting data
	update_funcs = {
			'count1': update_func,
			#'P1 count': update_func,
			'count2': update_func,
			#'P2 count': update_func,
			'count3': update_func,
			#'P3 count': update_func,
			'count4': update_func,
			#'P4 count': update_func,
			#'bins plot': update_func
	}
	

	make_movie.run(dr=dr, code=code, data_func=data_func, update_func=update_funcs, data=order,
	save_as=name, path=path, open=open_movie, time=time_range, dim='1d', **not_parsed) 
	



# if ran as a script
if (__name__ == '__main__'):
	run(1, [])