#!/usr/bin/python

import numpy as np
import os
import sys
import traceback
import time
import subprocess
import shutil
import utils.postproc_tools as tools
import utils.get_args as get_args
import utils.text_edit as txt
# there is a conditional import of mpi4py in run()
# so the user is not required to have mpi4py to use ppt serially


##### to-do #####
# - make commands() a separate module _commands.py
# - make executing a module a positional arg
#   if the posarg is not given, then _commands.py is the module called
#   include a list of possible moules in --help
# - create a single parser object and dr object in ppt.py
#   pass the objects to each subsequent module
#   - the parser "chain" should allow for --help, and __defaults__
#   - the chain should track the args at each level and the unparsed args at each level
#     so when it finally hits finish() it knows if bad args were given
# - include a ppt command for setdefaults, savedefaults, and cleardefaults
#   this iterates through each module and performs the respective act
# - could consider re-introducing a '.ppt' directory for temp/log files
#   - this would include storing defaults
#   - storing previous inputs
#   - storing input "sets" that run multiple modules at once

def run(module):
	"""Run one of the post-processing scripts with a high-level command"""
	parser = get_args.GetArgs(help_exit=False, help_chain=True)
	code       = parser.get('code', 'EPOCH', hide=True)
	# hide inputs that are used by ppt.run() and make_movie.run()
	path       = parser.get('dir', hide=True)
	run        = parser.get('run', hide=True)
	time_range = parser.get('time', hide=True)
	dump_range = parser.get('dump', hide=True)
	open_movie = parser.get('open', hide=True)
	# handle args that are unique to ppt.run()
	wait       = parser.get('wait', type=float, group='Wait', mutexc=True, help='Wait to run the given commands until a dump with time WAIT has been created')
	wait_dump  = parser.get('wait_dump', type=int, group='Wait', mutexc=True, help='Wait to run the given commands until a dump with number WAIT_DUMP has been created')
	use_mpi    = parser.get('mpi', False, help="Use an MPI implementation of ppt. Must use 'mpiexec -n NP ppt...")
	keep_logs  = parser.get('keep_logs', False, help='If True, keep all temporary files when using MPI')
	parser.finish(ignore=True)
	all_args = parser.get_all_args()

	rm_args = ['wait', 'wait_dump', 'mpi', 'keep_logs']
	if use_mpi is True:
		rm_args += ['time', 'dump', 'open']
	# remove args that are only used in ppt.py
	for arg in rm_args:
		try: del all_args[arg]
		except KeyError: pass	

	def create_arg_list():
		"""Create a list of args that can be given to a GetArgs object"""
		arg_list = []
		for arg, val in all_args.items():
			if arg == module.__name__: continue
			if isinstance(val, list) or isinstance(val, tuple):
				arg_list.append('{}={}'.format(arg, ','.join([str(v) for v in val])))
			elif val is True:
				arg_list.append(arg)
			else:
				arg_list.append('{}={}'.format(arg, val))
		return arg_list
	args = create_arg_list()

	# wait until the dump has been created
	if wait != None or wait_dump != None:
		# it may be required to wait for a dump to be created before a data reader can be made
		while True:
			# make data reader to find a dump given a time
			try: dr = tools.get_data_reader(code=code, path=path, run=run, time=time_range, dump=dump_range, quiet=True)
			except OSError: pass
			else: break
			time.sleep(1)
		# once a data reader exists, wait until the correct sdf file exists and is valid
		while True:
			dr.update()
			if wait != None:
				try: wait_dump = dr.get_dump(wait, tol=0.05)
				except ValueError: continue
			if dr.sdf_valid(wait_dump) is True:
				args.append('dump={}'.format(wait_dump))
				break
			time.sleep(1)

	# MPI implementation
	if use_mpi is True:
		from mpi4py import MPI
		from utils.mutex import Mutex

		comm = MPI.COMM_WORLD
		rank = comm.Get_rank()
		mutex = Mutex(comm)

		# handle all names and paths
		name = all_args.pop('name', 'test').replace('.mp4', '')
		dirname = os.path.join(tools.get_output_path(run), os.path.dirname(name))
		basename = os.path.basename(name)
		# determine the name of the temporary directory to store temporary files
		temp_dir = os.path.join(dirname, '.temp-{}'.format(basename))
		# add the .mp4 to name
		name = os.path.join(dirname, '{}.mp4'.format(basename))

		# create a temporary directory for each rank's movie
		movie_dir = os.path.join(temp_dir, 'movies')
		# each rank will make its own movie. As such, each rank has a unique name
		movie_name = os.path.join(movie_dir, '{}-{}'.format(basename, 'rank-{}'.format(rank)))
		all_args['name'] = movie_name

		# the name of the final combined log file
		log_name = os.path.join(dirname, '{}.log'.format(basename))
		# create a log file for each rank
		log_dir = os.path.join(temp_dir, 'logs')
		log_file = os.path.join(log_dir, 'rank-{}.log'.format(rank))

		# make directories
		if rank == 0:
			try: os.makedirs(log_dir)
			except OSError: pass # log_dir already exists
			try: os.makedirs(movie_dir)
			except OSError: pass # movie_dir already exists
			comm.barrier()
		# all other ranks wait until log dir has been made
		else:
			comm.barrier()			

		# each rank will make a movie of a number of dumps equal to total_dumps/num_ranks
		dr = tools.get_data_reader(code=code, path=path, run=run, time=time_range, dump=dump_range, quiet=True)
		dumps_per_rank = dr.num_dumps / comm.Get_size()
		remainder = dr.num_dumps % comm.Get_size()
		indices = []
		b = dr.first
		for i in range(comm.Get_size()):
			e = b + dumps_per_rank
			if i < remainder: e += 1
			indices.append((b,e))
			b = e
		ib, ie = indices[rank]
		all_args['dump'] = (ib, ie-1)

		# ensure that each rank does not try and open it's movie
		all_args['open'] = False
		# re-create the arg list with the rank-modified args
		args = create_arg_list()

		def finalization(rank=0, err=None):
			"""Construct a single log file from each rank's log file"""
			if rank == rank:
				sys.stdout = sys.__stdout__
				if err: print err
				# create combined log file
				with open(log_name, 'w') as f:
					if err:
						f.write('{}\n'.format(err))
					for log in sorted(os.listdir(log_dir)):
						f.write('{}\n'.format(txt.title(log)))
						try:
							logf = open(os.path.join(log_dir, log), 'r')
						except Exception as e:
							f.write('{}\n'.format(str(e)))
						else:
							f.write('{}\n'.format(''.join(logf.readlines())))				
				print 'created: {}'.format(log_name)

				# combine each individual movie into a single movie
				ffmpeg = tools.get_ffmpeg()
				# create a file for ffmpeg concatenation
				concat = os.path.join(movie_dir, 'concat.txt')
				with open(concat, 'w') as f:
					for file in sorted(os.listdir(movie_dir)):
						if not file.endswith('.mp4'): continue
						f.write("file '{}'\n".format(os.path.join(movie_dir, file)))
				
				# create log for concatenation process and call:
				# ffmpeg -y -f concat -safe 0 -i videolist.txt -c copy output.mp4
				with open(os.path.join(movie_dir, 'concat.log'), 'w') as f:
					subprocess.call(['ffmpeg', '-y', '-f', 'concat', '-safe', '0', '-i', concat, '-c', 'copy', name],
														stdout=f, stderr=subprocess.STDOUT)				
				print 'created: {}'.format(name)

				# handle temp files
				if keep_logs is True:
					print 'created movies: {}'.format(movie_dir)
					print 'created logs: {}'.format(log_dir)
				else:
					try:
						shutil.rmtree(temp_dir)
					except OSError as e:
						raise e

				if open_movie is True:
					tools.open_file(name)

		# redirect stdout to the log file: all print statements now go to log_file
		f = open(log_file, 'w', buffering=0)
		if rank == 0:
			sys.stdout = tools.Logger(f)
			print txt.title('mpi info')
			print 'dump distribution:'
			for i, ind in enumerate(indices):
				print '  rank-{}: {}'.format(i, ind)			
			print 'args: {}'.format(' '.join(args))
		else:
			sys.stdout = f
			print txt.title('mpi args')
			print 'args: {}'.format(' '.join(args))

		# attempt to run each rank
		try:
			module.run(2, args)
		except:
			# only allow the first rank to perform cleanup
			mutex.lock()
			finalization(rank=rank, err='Exception encountered in rank {}:\n{}'.format(
																								rank, traceback.format_exc()))
			comm.Abort()

		# all ranks successfully made their movie, finalize the log file and movie
		comm.Barrier()
		if rank == 0:
			finalization()

	# Serial Implementation
	else:
		module.run(2, args)

# get_args outside of function commands(), use a loop to read how many get()'s are called for diagnostic scripts
# read all commands into a list and pass them into commands() so only 1 command is executed at once
# default commands if a script isn't specified
def commands():
	parser = get_args.GetArgs(help='Read data files and run diagnostics.', show_args=True)
	path          = parser.get('dir', help='<path> to a directory to read data from')
	run           = parser.get('run', type=str)
	code          = parser.get('code', 'EPOCH', help='The name of the code used to create data.')
	show          = parser.get('show', False, help='display data types in a .sdf file')
	get           = parser.get('get', help='display the given <datatype>')
	get_dump      = parser.get('get_dump', help='Given a time, get the corresponding dump')
	dump          = parser.get('dump', help='the <number> of a specific .sdf file')
	data_type_max = parser.get('max', help='return the max value of a given <datatype>')
	data_type_min = parser.get('min', help='return the min value of a given <datatype>')
	to_si         = parser.get('tosi', type=list, help='Given DATATYPE,DATA convert DATA from dimensionless to si units')
	to_dimless    = parser.get('todimless', type=list, help='Given DATATYPE,DATA convert DATA from si to dimensionless units. DATA must be given in standard S.I. units (i.e., meters not millimeters)')
	si            = parser.get('si', False, help='return data in s.i. units')
	parser.finish()

	dr = tools.get_data_reader(code=code, path=path, run=run)
	# set the default dump val to the first available dump number
	if dump is None:
		dump = dr.first
	# display the data types in a data dump
	if show:
		dr.show_data_types()

	# get a piece of data and display it
	if get:
		if isinstance(get, list):
			for data_type in get:
				dr.display(data_type, dump=dump, si=si)
		else:
			dr.display(get, dump=dump, si=si)

	if get_dump:
		if isinstance(get_dump, list):
			for time in get_dump:
				print 'time {} = dump {}'.format(time, dr.get_dump(time))
		else:
			print 'time {} = dump {}'.format(get_dump, dr.get_dump(get_dump))

	# define a function that gets a max or min
	def get_max_min(data_type, min_max, f):
		if data_type:
			if isinstance(data_type, list):
				for d in data_type:
					get_max_min(d, min_max, f)
			else:
				data = dr.get(data_type, dump=dump, si=si)
				print data_type,min_max,'=',f(data)

	# find the max for a given data type
	get_max_min(data_type_max, 'max', np.amax)

	# find the min for a given data type
	get_max_min(data_type_min, 'min', np.amin)

	if to_si:
		data_type = to_si.pop(0)
		for d in to_si:
			dr.to_si(data_type, d)
	if to_dimless:
		data_type = to_dimless.pop(0)
		for d in to_dimless:
			dr.to_dimless(data_type, d)

##########################   post-processing tools functionality   ##################################
# ppt will search the directory it is in to see if there are other python scripts
# if the first input given is the name of a valid python script, ppt will run that script 
# and pass the remaining user inputs to that script
# if the first input is not a valid python script, then the inputs will be given to commands()

# path that this file is in
script_dir = tools.get_script_path()
# get a list of all python scripts in the directory
script_list = []
for file in os.listdir(script_dir):
	
	# check the file extension if it exists
	try:
		filename, extension = file.split(".", 1)
	# file doesn't have an extension. ignore.
	except:		
		pass
	# add python files to script_list
	else:
		if extension == "py":
			script_list.append(filename)

# if the user gives no input arguments, then display possible script names
if len(sys.argv) == 1:
	print 'Possible Scripts:'
	for s in script_list:
		print '  {}'.format(s)
else:
	# the first user argument is the name of a script to run
	script = sys.argv[1]
	# allow the script to be given with '.py' appended
	script = script.replace('.py', '')
	if script in script_list:
		mod = __import__(script)
		try:
			#mod.run(2, code='EPOCH')
			run(mod)
		except Exception as e:
			traceback.print_exc()
	# if the first argument is not a script name then it is a command
	else:
		commands()

