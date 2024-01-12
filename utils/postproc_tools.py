#!/usr/local/bin/python

import numpy as np
from scipy import constants as sc 
import sys
import os
import inspect
import socket
import platform
import subprocess
from timeit import default_timer as timer
import psutil
import utils.text_edit as txt


##################################   Functions   ##########################################
def output_progress(output_str, t, begin, end, skip=1):
	"""Output the progress to stdout given a label: 'output_str', a timestep 't', a
	beginning timestep 'begin', and an ending timestep 'end'."""
	output = '%s... %0.1f%%' % (output_str, ((float(t) - begin) / (float(end) - begin)) * 100)
	sys.stdout.write("\r" + output)
	sys.stdout.flush()
	# reset for next output
	# end-1 because this will be called in a loop and range(begin, end)
	# is not inlusive
	if (t == end-skip):
		output = '%s... 100.0%%' % output_str
		sys.stdout.write("\r" + output)
		sys.stdout.flush()
		print ''

def get_os():	
	"""Get the name of the operating system"""
	return platform.system()

def get_home():
	"""Return the environment variable $HOME"""
	return os.environ["HOME"]

def get_script_path():
	"""Return the path to the tools directory"""
	return os.path.dirname(os.path.dirname(os.path.realpath(inspect.getfile(inspect.currentframe()))))

def get_analysis_path():
	"""Checks to see if the UNL directory structure is being used. If so, it returns the
	path to '/EPOCH/analysis/'. Otherwise it returns None.
	"""
	# allow an environment variable to set the location
	if 'EPOCH_ROOT' in os.environ:
		return os.path.join(os.environ['EPOCH_ROOT'], 'analysis')
	# check if the UNL directory structure is being used: /EPOCH/scripts/tools/
	path = os.path.dirname(os.path.dirname(get_script_path()))
	if os.path.basename(path) != 'EPOCH': return None
	else:
		# check if: /EPOCH/analysis exists
		if 'analysis' in os.listdir(path):
			return os.path.join(path, 'analysis')
		else: return None

def get_data_path():
	"""Checks to see if the UNL directory structure is being used. If so, it returns the
	path to '/EPOCH/data/'. Otherwise it returns None.
	"""
	# allow an environment variable to set the location
	if 'EPOCH_ROOT' in os.environ:
		return os.path.join(os.environ['EPOCH_ROOT'], 'data')
	# check if the UNL directory structure is being used: /EPOCH/scripts/tools/
	path = os.path.dirname(os.path.dirname(get_script_path()))
	if os.path.basename(path) != 'EPOCH': return None
	else:
		# check if: /EPOCH/data exists
		if 'data' in os.listdir(path):
			return os.path.join(path, 'data')
		else: return None

def get_output_path(*run):
	"""Return the path where the output of a script should be saved.
	If the UNL directory structure is being used and a RUN is given it will return '/EPOCH/analysis/RUN'.
	If the environment variable 'PPT_OUTPUT' is set, that path will be used.
	Otherwise, 'postproc_tools/output' is returned.
	"""
	# allow an environment variable to set the location
	if 'PPT_OUTPUT' in os.environ:
		return os.environ['PPT_OUTPUT']
	if len(run) > 0:
		if run[0] != None:
			path = get_analysis_path()
			if path: return os.path.join(path, run[0])
	# if run is None or not given
	return os.path.join(get_script_path(), 'output')

def get_queue_path():
	"""Checks to see if the UNL directory structure is being used. If so, it returns the
	path to '/EPOCH/scripts/queue'. Otherwise it returns None.
	"""
	# allow an environment variable to set the location
	if 'queue' in os.environ:
		return os.environ['queue']
	elif 'EPOCH_ROOT' in os.environ:
		return os.path.join(os.environ['EPOCH_ROOT'], 'scripts/queue')
	# check if the UNL directory structure is being used: /EPOCH/scripts/tools/
	path = os.path.dirname(os.path.dirname(get_script_path()))
	if os.path.basename(path) != 'EPOCH': return None
	else:
		# check if: /EPOCH/data exists
		if 'scripts' in os.listdir(path):
			if 'queue' in os.listdir(os.path.join(path, 'scripts')):
				return os.path.join(path, 'scripts/queue')
		return None

def to_queue(command, *commands):
	"""Given a command as a list of args, write them to the queue.
	Additional commands can be given as optional arguments.
	"""
	all_commands = [command]
	all_commands.extend(commands)
	# submit the commands to the queue
	with open(os.path.join(get_queue_path(), 'commands.txt'), 'a') as f:
		for c in all_commands:
			c = ' '.join(c)
			f.write('{}\n'.format(c))
			print 'submitted: {}'.format(c)

def is_crane():
	"""Check to see if the script is being run on the Crane supercomputing cluster"""
	if 'crane' in socket.gethostname():
		return True
	else:
		return False

def is_swan():
	"""Check to see if the script is being run on the Swan supercomputing cluster"""
	if 'swan' in socket.gethostname():
		return True
	else:
		return False

def is_mlviz():
	"""Check to see if the script is being run on the Meadowlark cluster"""
	if socket.gethostname() == 'ml-viz':
		return True
	else:
		return False

def get_hostname():
	"""Get the hostname of the machine"""
	return socket.gethostname()

def get_ffmpeg():
	"""Search for the location of ffmpeg and return its path"""
	# search "/usr/bin"
	path = '/usr/bin'
	for file in os.listdir(path):
		if file == 'ffmpeg':
			return os.path.join(path, 'ffmpeg')

	# search "/usr/local/bin"
	path = '/usr/local/bin'
	for file in os.listdir(path):
		if file == 'ffmpeg':
			return os.path.join(path, 'ffmpeg')

	# search each conda environment to see if it has ffmpeg
#	path = os.path.join(os.environ['HOME'], '.conda/envs')
#	if os.path.exists(path):
#		for env in os.listdir(path):
#			env = os.path.join(path, env, 'bin')
#			for file in os.listdir(env):
#				if file == 'ffmpeg':
#					return os.path.join(env, 'ffmpeg')
	conda_envs_dir = os.path.join(os.environ['HOME'], '.conda/envs')
	for root, dirs, files in os.walk(conda_envs_dir):
		for file in files:
			if file == 'ffmpeg':
				return os.path.join(root, file)

	# ffmpeg path could not be found, raise error
	raise OSError(('Could not find ffmpeg. Checked /usr/bin/, /usr/local/bin/, and conda environments. '
								 'If ffmpeg is in another location, try hardcoding in the path as a return statement '
								 'in the function utils.postproc_tools.get_ffmpeg()'))

image_format = 'png'
_possible_formats = ['ps', 'eps', 'pdf', 'pgf', 'png', 'raw', 'rgba', 'svg', 'svgz', 'jpg', 'jpeg', 'tif', 'tiff']
def save_image(fig, path, dpi=300, format=image_format, **kwargs):
	"""Save the image of the matplotlib figure FIG at PATH.
	If PATH ends with a file extension, that image made will be of that file type.
	If PATH does not have a file extension, use postproc_tools.image_format
	All kwargs are passed to the matplotlib figure.savefig() call.
	"""
	# set extension if need be
	for f in _possible_formats:
		if path.endswith('.{}'.format(f)):
			break
	# path does not have an extension, add it
	else:
		path = '{}.{}'.format(path, format)

	# make output dir
	try: os.makedirs(os.path.dirname(path))
	# dir already exists	
	except OSError: pass
	fig.savefig(path, dpi=dpi, **kwargs)
	print 'image saved: {}'.format(path)
	return path

def open_file(path):
	"""Open a file with host-specific instructions."""
	print 'path: {}'.format(path)
	if get_os() == 'Linux':
		hostname = get_hostname()
		if hostname == 'pc-linux' or hostname == 'laptop-linux':
			with open(os.devnull, 'w') as quiet:
				subprocess.call(['xdg-open', path], stdout=quiet, stderr=quiet)
		elif is_crane() is True:
			pass
		elif is_swan() is True:
			pass
		elif hostname == 'ml-viz':
			pass
		else:
			# open on windows subsystem for linux
			subprocess.call(['wsl_open', path])
	else:
		# open on MacOS
		subprocess.call(['open', path])


# allow readers to be imported
codes = {
	'EPOCH': '.sdf',
	'Smilei': '.h5'
}

def get_data_reader(code=None, path=None, **kwargs):
	"""Get the appropriate data reader from 'postproc_tools/readers'
	All additional kwargs are passed to the data reader constructor
	"""
	# need either code or path to be given
	if code is None and path is None:
		errmsg = 'Must give the code, path, or both'
		raise TypeError(errmsg)

	# if a path is given, check to see if the code name can be determined from file types
	if code is None:
		for file in os.listdir(path):
			# check each code name and its file type
			for key in codes:
				if file.endswith(codes[key]):
					code = key
					break
			if code:
				break
		if code is None:
			file_type_str = ''
			for key in codes:
				file_type_str += ' %s %s' % (key, codes[key])
			errmsg = 'Could not find valid file types (%s) in: %s' % (file_type_str, path)
			raise IOError(errmsg)

	if code == 'EPOCH':
		import readers.sdf_reader as sdf_reader
		return sdf_reader.SDF_Reader(code=code, path=path, **kwargs)
	elif code == 'Smilei':
		import readers.smilei_reader as smilei_reader
		return smilei_reader.Smilei_Reader(code=code, path=path, **kwargs)
	else:
		errmsg = "%s is not a supported code. Try 'EPOCH' or 'Smilei'." % code
		raise TypeError(errmsg)

##################################   Logger   #########################################
class Logger(object):
	"""Allow stdout to be directed to both a log file and console"""
	def __init__(self, file):
		self._console = sys.stdout
		self._log = file

	def write(self, message):
		"""Write to both the console and log file"""
		self._console.write(message)
		self._log.write(message)

	def flush(self):
		"""Call to sys.stdout.flush()"""
		self._console.flush()
		self._log.flush()

##################################   Timer   ##########################################
class Timer(object):
	"""Create a Timer object that uses the default_timer from timeit."""
	def __init__(self, **kwargs):
		"""Automatically start the timer."""
		self.name = kwargs.get('name', 'diagnostics')
		self._process = psutil.Process(os.getpid())
		self._timer = timer
		self._start_time = 0.
		# non-swapped physical memory used by the process (RSS)
		self._start_mem = 0.
		self._max_mem = 0.
		# virtual memory used by the process
		self._start_vmem = 0.
		self._max_vmem = 0.
		# track snapshots of the current process
		self.snapshots = {
			'name': [],
			'time': [],
			'mem' : [],
			'vmem': []
		}
		self.start()

	def start(self, *args):
		"""Set the start time.
		A string can be passed as an optional argument that will display when the timer starts.
		"""
		if len(args) == 1:
			print args[0]
		self._start_mem = self.get_mem()
		self._start_vmem = self.get_vmem()
		self._start_time = self._timer()

	def get_time(self):
		"""Return the time in seconds since the timer started."""
		return self._timer() - self._start_time

	def stop(self, *args, **kwargs):
		"""Return display string with the runtime and memory usage since starting.
		Set the stop time and display the time that has passed from the point the timer was started.
		A string can be passed as an optional argument that will display along with the time passed.
		Possible Kwargs
			full: If true, display all memory info
			mem: If true, the memory usage will be displayed in Mb
			vmem: If true, the virtual memory usage will be displayed in Mb
			all: If true, all other diagnostic types are true
			display: If true, return the the displayed string. Default True		
		"""
		full = kwargs.get('full', False)
		use_all = kwargs.get('all', False)
		end_time = self._timer()
		runtime = end_time - self._start_time
		if runtime > 60:
			m, s = divmod(runtime, 60)
			h, m = divmod(m, 60)
			if len(args) == 1:
				s = '%s: %d:%02d:%02d' % (args[0], h, m, s)
			else:
				s = 'runtime: %d:%02d:%02d' % (h, m, s)
		else:
			if len(args) == 1:
				s = '{}: {} seconds'.format(args[0], runtime)
			else:
				s = 'runtime: {} seconds'.format(runtime)

		if kwargs.get('mem', False) or use_all:
			s = '{}\n{}'.format(s, self.memory(display=False, full=full))

		if kwargs.get('vmem', False) or use_all:
			s = '{}\n{}'.format(s, self.vmemory(display=False, full=full))
		
		# return runtime
		if kwargs.get('display', True): 
			print s
		return s

	def get_mem(self, **kwargs):
		"""Get the current memory usage of the process.
		Possible Kwargs:
		  start: If True, return the memory usage since created or start() called
		"""
		mem = self._process.memory_info().rss / 1.e6
		if kwargs.get('start', False):
			return mem - self._start_mem
		else:
			return mem

	def get_vmem(self, **kwargs):
		"""Get the current virtual memory usage of the process.
		Possible Kwargs:
		  start: If True, return the virtual memory usage since created or start() called
		"""
		vmem = self._process.memory_info().vms / 1.e6
		if kwargs.get('start', False):
			return vmem - self._start_vmem
		else:
			return vmem

	def snapshot(self, *args):
		"""Add the current memory and time to snapshots a dict of lists with keys (name, time, mem, vmem).
		If the memory exceeds the current max usage, set the new maximum.
		An arg can be given as a string to "name" the snapshot.
		"""
		# determine snapshot name
		if len(args) == 1:
			name = args[0]
		else:
			name = 'snapshot {}'.format(len(self.snapshots))
		self.snapshots['name'].append(name)
		self.snapshots['time'].append(self.get_time())
		self.snapshots['mem'].append(self.get_mem())
		self.snapshots['vmem'].append(self.get_vmem())

	def _get_mem_str(self, mtype, *args, **kwargs):
		"""Create the memory string based on the memory type
		Possible kwargs:
		  start: If True, return the memory usage since created or start() called
		  current: If True, display the current usage. Default True.
		  full: Display the maximum memory, average memory, and memory snapshots. Note, this is the 
		        max and avg of the snapshots taken.
		  snapshots: If True, display the snapshots. Default False
		"""
		s = []
		if kwargs.get('current', True):
			if mtype == 'mem': mem = self.get_mem(**kwargs)
			elif mtype == 'vmem': mem = self.get_vmem(**kwargs)
			if len(args) == 1:
				s.append('{}: {}Mb'.format(args[0], mem))
			else:
				s.append('{}: {}Mb'.format(mtype, mem))

		if kwargs.get('full', False):
			s.append('max {}: {}Mb'.format(mtype, max(self.snapshots[mtype])))
			avg = float(sum(self.snapshots[mtype])) / len(self.snapshots[mtype])
			s.append('avg {}: {}Mb'.format(mtype, avg))
			if kwargs.get('snapshots', False):
				for m in self.snapshots[mtype]:
					s.append('  {}: {}Mb'.format(s, m[0], m[1]))
		return '\n'.join(s)

	def memory(self, *args, **kwargs):
		"""Display the memory usage in Mb of the entire Python process
		A string can be passed as an optional argument that will display along with the usage.
		Possible kwargs:
		  start: If True, return the usage since created or start() called
		  current: If True, display the current usage. Default True.
		  display: If true, return the the displayed string. Default True
		  full: Display the maximum usage and average usage. Note, this is the 
		        max and avg of the snapshots taken.
		  snapshots: If True, display the snapshots. Default False
		"""
		s = self._get_mem_str('mem', *args, **kwargs)
		if kwargs.get('display', True):	
			print s
		return s

	def vmemory(self, *args, **kwargs):
		"""Display the virtual memory usage in Mb of the entire Python process
		A string can be passed as an optional argument that will display along with the usage.
		Possible kwargs:
		  start: If True, return the usage since created or start() called
		  current: If True, display the current usage. Default True.
		  display: If true, return the the displayed string. Default True
		  full: Display the maximum usage and average usage. Note, this is the 
		        max and avg of the snapshots taken.
		  snapshots: If True, display the snapshots. Default False
		"""
		s = self._get_mem_str('vmem', *args, **kwargs)
		if kwargs.get('display', True):	
			print s
		return s

	def display(self, **kwargs):
		"""Stop the timer and display a formatted output.
		Possible Kwargs:
		  pid: If true, display the process id
		  args: If true, display the input args. Default True
		"""
		s = []
		s.append(txt.divider())		
		s.append(self.name)
		if kwargs.get('pid', False):
			s.append('pid: {}'.format(self._process.pid))
		if kwargs.get('args', True):
			s.append(' '.join(sys.argv[1:]))
		s.append(self.stop(display=False))
		s.append(self.memory(display=False, full=True, current=False))
		s.append(self.vmemory(display=False, full=True, current=False))
		s.append(txt.divider())
		print '\n'.join(s)
