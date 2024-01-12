import sys
import os
import inspect
import text_edit as txt

###############################   Option Class   ########################################
class Option(object):
	"""Class used to model and store information for a command line argument."""
	def __init__(self, name, 
							alt_names=[], 
							help=None,
							val=None,
							default=None,
							is_given=False,
							hide=False):
		self.name = name
		self.alt_names = alt_names
		# ensure alt_names is a list
		if not isinstance(self.alt_names, list):
			self.alt_names = [self.alt_names]
		self.help = help
		self._val = val
		self._default = default
		self._is_given = is_given
		self.hide = hide

	def _is_arg(self, arg):
		"""Used to determine if a given string is equal to name or alt_name"""
		if arg == self.name:
			return True
		for alt in self.alt_names:
			if arg == alt:
				return True
		# arg is not equal to name or alt_name
		return False

	def get_val(self):
		return self._val

	def is_given(self):
		"""Returns true if the option has been given as an arg."""
		return self._is_given

	def get_help_str(self, pad, show_defaults=True):
		help_str = '  {}'.format(self.name)
		if self.alt_names:
			help_str += ', '
			help_str += ', '.join(self.alt_names)
		help_str = help_str.ljust(pad)
		if self.help:
			help_str += self.help
		if self._default and show_defaults is True:
			if not help_str.endswith('.'):
				help_str += '.'
			if isinstance(self._default, list):
				if len(self._default) == 1: default_str = self._default[0]
				else: default_str = ','.join([str(d) for d in self._default])
			else: default_str = self._default
			help_str += ' Default: {}'.format(default_str)
		return txt.wrap(help_str, indent=pad)

	def display(self, pad, show_defaults=True):
		# display_str = '  {}'.format(self.name)
		# if self.alt_name:
		# 	display_str = display_str + ', {}'.format(self.alt_name)
		# display_str = display_str.ljust(pad)
		# if self.help:
		# 	display_str = display_str + self.help
		# if self._default and show_defaults is True:
		# 	if not display_str.endswith('.'):
		# 		display_str += '.'
		# 	display_str += ' Default: {}'.format(self._default)
		# print display_str
		print self.get_help_str(pad, show_defaults=show_defaults)


###############################   Group Class   ########################################
class Group(list):
	"""A collection of command line arguments."""
	def __init__(self, name, 
							required=False,
							help=None,
							mutually_exclusive=False,
							option=None):
		super(Group, self).__init__()
		self.name = name
		# determines if args from this Group are required
		self._required = required
		# description for the group
		self._help = help
		# determines if the group is mutually exclusive
		self._mutually_exclusive = mutually_exclusive
		# maximum length of any Option name within Group (default = 20)
		self._max_name_length = 17
		# append an Option from kwargs
		temp_option = option
		if temp_option:
			self.append(temp_option)
		# index for iterator
		self._index = 0

	def _has_option(self, option):
		"""Check the group's options, if option exists return True, otherwise return False"""
		for o in self:
			if o._is_arg(option):
				return True
		return False

	def _get_option(self, option):
		"""Given an option name, return that option, else return None"""
		for o in self:
			if o._is_arg(option):
				return o
		raise KeyError("Option '{}' does not exist in Group '{}'".format(option, self.name))

	def get_inputs(self):
		"""Return a dictionary of all options with key-value pairs (option_name, option_value)."""
		d = {}
		for o in self:
			d[o.name] = o.get_val()
		return d

	def hide(self):
		"""If all options are hidden, return True"""
		for option in self:
			if option.hide is False:
				return False
		return True

	def append(self, option):
		"""Append a new option to the group. This overrides the append() function of the list class"""
		# get the name length (name, alt_names) of the new option
		name_length = len(option.name)
		for alt in option.alt_names:
			name_length += 2 + len(alt)
		# add 3 spaces to the maximum name_length for formatting
		name_length += 3
		# set new max_name_length if necessary
		if name_length > self._max_name_length:
			self._max_name_length = name_length
		super(Group, self).append(option)
		self.ensure_mutually_exclusive()

	def sort(self):
		"""Sort all options alphabetically."""
		super(Group, self).sort(key=lambda x: x.name)
		# move the '--help' option to the back of the list
		i = next((x for x in self if x.name == '--help'), None)
		if i:
			self.append(self.pop(self.index(i)))

	def ensure_mutually_exclusive(self):
		"""If the group is mutually exclusive, ensure only 1 option is given."""
		if self._mutually_exclusive is True:
			count = 0
			for option in self:
				if option.is_given() is True:
					count += 1
			if count > 1:
				print 'multiple options given in mutually exclusive group:'
				self.display()
				sys.exit()

	def usage(self):
		"""Return a string for 'Usage:' in --help"""
		# return an empty string if there are no options in the group
		if len(self) == 0:
			return ''
		# do not include '[]' if an option is required
		if self._required is True:
			help_str = self.name.upper()
		else:
			help_str = '[{}]'.format(self.name.upper())
		# include '...' if there can be multiple options given
		if len(self) > 1 and self._mutually_exclusive is False:
			help_str = help_str + '...'
		return help_str

	def get_help_str(self, show_defaults=True, alt_name=None):
		"""Create a string that holds the information of the group's options."""
		# do not display any information if there are no Options
		if len(self) == 0:
			return ''
		# include a new line before each new group
		if alt_name:
			help_str = alt_name
		else:
			help_str = self.name
		if self._mutually_exclusive is True:
			help_str += ' (Mutually Exclusive)'
		# if a description has been given, display that, otherwise display the name
		if self._help:
			help_str += ' - {}:'.format(self._help)
		else:
			help_str += ':'
		self.sort()
		lines = [help_str]
		for option in self:
			if option.hide is True: continue
			lines.append(option.get_help_str(self._max_name_length, show_defaults=show_defaults))
		help_str = '\n'.join(lines)
		return help_str

	def display(self, show_defaults=True):
		"""Display the information of the group's options."""
		print self.get_help_str(show_defaults=show_defaults)

	def get_option_help_str(self, option, show_defaults=True):
		"""Return an option's help string. Raises a KeyError if that option could not be found"""
		o = self._get_option(option)
		pad = len(o.name)
		for n in o.alt_names:
			pad += len(n)
		return o.get_help_str(pad + 7)

	def display_option(self, option, show_defaults=True):
		"""Display an option. Raises a KeyError if that option could not be found"""
		print self.get_option_help_str(option, show_defaults=show_defaults)



###############################   GetArgs Class   #######################################
class GetArgs(object):
	"""Class used in order to parse command line arguments and return them as variables."""
	def __init__(self, 
							args=[],
							start=1,
							posargs=[],
							max_posargs=None,
							required_posargs=None,
							auto_type=True,
							app_name=None,
							help=None,
							help_exit=True,
							help_chain=False,
							show_defaults=True,
							show_args=False,
							help_args=[],
							overrides={}):
		"""If *args is given, either as multiple arguments or a single list of arguments, then
		those args will be used instead of sys.argv
		"""
		# handle args
		if len(args) == 1:
			args = args[0]
		if isinstance(args, tuple):
			args = list(args)
		if not isinstance(args, list):
			args = [args]

		# list that stores the inputs that have been parsed
		self._arg_list = []
		# index of sys.argv to start reading inputs at if args is not given
		if args:
			self._inputs = self.create_input_dict(args)
			self._raw_inputs = args
		else:
			start = start		
			self._inputs = self.create_input_dict(sys.argv[start:])
			self._raw_inputs = sys.argv[start:]

		# a list of the posarg names for the usage
		self._posargs_names = posargs
		if not isinstance(self._posargs_names, list):
			self._posargs_names = [self._posargs_names]
		# the nmaximum number of posargs that can be given
		self._num_posargs = max_posargs
		if self._num_posargs is None:
			try: self._num_posargs = len(self._posargs_names)
			except: self._num_posargs = None
		# the number of required posargs
		self._num_required_posargs = required_posargs	
		self._posargs = []
		self._auto_type = auto_type	
		# the name of the application for --help
		# if not given, the name will be found via get_source_info()
		self._app_name = app_name
		# a user defined description for --help
		self._help = help
		# if True, exit when a help arg is given
		self._help_exit = help_exit
		# if True, then this parser is the first in a help chain
		# change any help_arg to '--help-options-only'
		self._help_chain = help_chain
		# determine if default values will be shown if '--help' is given
		self._show_defaults = show_defaults
		# if true, show the parsed args once finished
		self._show_args = show_args
		# determine the name of __main__ and the number of inputs a user asks for
		# self._app_name, self._source_code, self._num_inputs are defined here
		self.get_source_info()
		# a list of args that trigger the usage display behavior
		self._help_args = ['-h', '--help']
		# a string or list of strings for arg names that trigger the help usage behavior
		if not isinstance(help_args, list):
			help_args = [help_args]
		self._help_args.extend(help_args)
		# a variable to handle the '__help_min' flag which is used to display only self._help
		self._help_min = False		
		# a dictionary that will be checked after the self._inputs dictionary.
		# if a value exists in both inputs and overrides, the value in overrides will be used
		self._overrides = overrides	
		# the number of inputs that have been parsed
		self._num_parsed = 0
		# a list of acceptable types that a user can specify an input to be parsed to
		self._acceptable_types = [str, int, float, list, tuple, bool]
		# a list of types that a elements of a list/tuple cannot be parsed to
		self._illegal_list_types = [list, tuple]
		# create a list of alt_names for the --help option
		temp_help_args = self._help_args[:]
		temp_help_args.remove('--help')
		# dictionary that stores Groups of options
		self.groups = {
			# default group for storing required arguments
			'Required': Group('Required', required=True),
			# default group for storing optional arguments
			'Default': Group('Option', option=Option('--help', alt_names=temp_help_args, help='display this page'))
		}		

	# get information through the stack
	# determine the name of the calling script
	# determine the number of inputs by checking the number of times self.get() is called
	def get_source_info(self):
		# get the app_name from the initial file in the stack call
		if self._app_name is None:
			self._app_name = os.path.basename(inspect.stack()[2][1])

		# use the interpreter stack's frame record to get the frame in which this object was created
		frame = inspect.stack()[2]
		# if a script creates a GetArgs object, it is said to be a module
		if frame[3] == '<module>':
			self._source_code = inspect.getsourcelines(inspect.getmodule(frame[0]))
		# get the source code of the function that created this object (self)
		else:			
			self._source_code = inspect.getsourcelines(frame[0])
		# get the name of the variable that references this object (self)
		obj_name = None
		for line in self._source_code[0]:
			if 'GetArgs(' in line:
				obj_name = line.split('=', 1)[0].strip()
				break
		# determine the number of times a user calls self.get()
		self.num_args = 0
		get_str = obj_name + '.get('
		finish_str = obj_name + '.finish('
		for line in self._source_code[0]:
			line = line.strip()
			if line.startswith('#'): continue
			if get_str in line:
				self.num_args += 1
			elif finish_str in line:
				# increase num_args above the number of .get() calls so .finish() is called
				self.num_args += 1

	# parse a list of command line inputs to a dictionary
	def create_input_dict(self, input_args):
		# # remove the positional arguments
		# if self._num_posargs > 0:
		# 	for i in range(len(input_args)-1, -1, -1):
		# 		if input_args[i].startswith('-'): continue
		# 		self._posargs.append(input_args.pop(i))
		# 		if len(self._posargs) == self._num_posargs: break
		# 	else:
		# 		if '-h' not in input_args and '--help' not in input_args:
		# 			print "expecting '{}' positional arguments. Got '{}'".format(self._num_posargs, len(self._posargs))
		# 			print 'try -h for more help'
		# 			sys.exit()
		
		inputs = {}
		for arg in input_args:
			if arg == '__help_min':
				self._help_min = True
				continue
			if "=" in arg:
				temp = arg.split("=", 1)
				name = temp[0].strip()
				val = temp[1].strip()
				#val = self.get_type(val)
				inputs[name] = val
				self._arg_list.append(name)
			else:
				inputs[arg] = True
				self._arg_list.append(arg)
		return inputs

	# return the input as it was given
	def get_raw_input(self, name):
		if self._inputs[name] == name:
			return name
		else:
			return '{}={}'.format(name, self._inputs[name])

	# idetermine the type of the data
	def get_type(self, val, **kwargs):
		# if parsing a list, the user can specify the type of delimiter
		delimiter = kwargs.get('delimiter', ',')

		if not isinstance(val, str):
			return val
			
		# check to see if an input is multi-valued
		if delimiter in val:
			temp = val.split(delimiter)
			val_list = []
			for t in temp:
				val_list.append(self.get_type(t))
			return val_list
		else:
			# attempt to parse into an integer
			# if the input is a float it will throw an exception
			try:
				val = int(val)
			except:
				# check to see if val is a boolean
				if val.lower() == 't' or val.lower() == 'true':
					val = True
				elif val.lower() == 'f' or val.lower() == 'false':
					val = False
				# check to see if val is 'None'
				if val == 'None':
					val = None
				# check to see if val is a double
				elif '.' in val or 'e' in val:
					# attempt to parse into a double
					try:
						val = float(val)
					except:
						# val is a string with a '.' or 'e' in it
						pass
			finally:
				return val

	def parse_arg(self, name, default, **kwargs):
		# an alternate name for the input
		alt_names = kwargs.get('alt_name', [])
		alt_names = kwargs.get('alt_names', alt_names)		
		# if required is true, then the sys.exit() will be called if the input has no value
		required = kwargs.get('required', False)
		# only check overrides if override is true
		override = kwargs.get('override', True)
		# attempt to parse input to a specific type
		arg_type = kwargs.get('type')
		# if type is a list or tuple, attempt to parse all elements to ltype
		list_type = kwargs.get('ltype')
		# a required length for lists or tuples can be given
		list_length = kwargs.get('len')

		# an alt_name can also be given as a string. Ensure alt_names is a list
		if not isinstance(alt_names, list):
			alt_names = [alt_names]
		# ensure that the name is not the protected '--help'
		if any(arg == '--help' for arg in [name] + alt_names):
			print "arg '--help' is protected and cannot be used"
			sys.exit(2)	
		# if -h is used as an argument, remove it as an equivalent to '--help'
		if any(arg == '-h' for arg in [name] + alt_names):
			try: self._help_args.remove('-h')
			except ValueError: pass # '-h' is already removed from _help_args
			# remove the alt_name '-h' from the '--help' option
			try: self.groups['Default']._get_option('--help').alt_names.remove('-h')
			except KeyError: pass   # '--help' option is already removed
			except ValueError: pass # '-h' is already removed as an alt_name

		# if input is given explicitly as None 
		# or no input is given (i.e., arg_name= ), return None
		none_check = self._inputs.get(name, 1)
		if none_check == 'None' or none_check == '':
			return None

		# check inputs for arg
		val = self._inputs.get(name)
		try: raw_val = self.get_raw_input(name)
		except: raw_val = '{}={}'.format(name, val)	
		if val is None:
			for alt in alt_names:
				val = self._inputs.get(alt)
				try: raw_val = self.get_raw_input(alt)
				except: raw_val = '{}={}'.format(alt, val)
				if val: break

		def change_type(v, t=arg_type, errstr=raw_val):
				"""Attempt to return V as type T.
				If T is None, return V as it was given.
				Exit if T(v) raises a ValueError."""
				if t is None: return v
				# parse an int to a float first to account for scientific notation
				if arg_type == int:
					try:
						v = float(v)
					except ValueError:
						if errstr == raw_val:
							errstr = "'{}'".format(errstr)
						print "{} cannot be parsed to type {}".format(errstr, t)
						sys.exit(2)
				# attempt to change v to type t and raise error if necessary
				try:
					v = t(v)
				except ValueError:
					if errstr == raw_val:
						errstr = "'{}'".format(errstr)
					print "{} cannot be parsed to type {}".format(errstr, t)
					sys.exit(2)
				else:
					return v

		def assert_list(l):
			"""If LIST_LENGTH is defined, assert that L is of that length.
			If LIST_TYPE is defined, assert that each element in L is of LIST_TYPE.
			L is returned if no errors are encountered, otherwise exit the application.
			"""
			if list_length and len(l) != list_length:
					print "'{}' must be {} of length: {}".format(raw_val, arg_type, list_length)
					sys.exit(2)
			if list_type:
				if arg_type is tuple or isinstance(l, tuple): l = list(l)
				if list_type in self._illegal_list_types:
					legal_types = [str(t).replace('<type ','').replace('>', '') for t in self._acceptable_types if t not in self._illegal_list_types]
					legal_types = ', '.join(legal_types)
					print "cannot change the elements in '{}' to: {}. Try: {}".format(name, list_type, legal_types)
					sys.exit(2)
				for i, v in enumerate(l[:]):
					l[i] = change_type(v, t=list_type, errstr="element in '{}': '{}'".format(name, v))
			if arg_type is tuple and isinstance(l, list): l = tuple(l)
			return l

		# check if a kwarg override exists
		if isinstance(override, bool):
			if override is True:
				for alt in alt_names:
					val = self._overrides.get(alt, val)
				# name takes precedence over alt_name
				val = self._overrides.get(name, val)
		else:
			raise TypeError('get() keyword "override" expecting an input of type boolean')

		# check if val has a default value and exit if necessary
		if val is None:
			val = default
			raw_val = '{}={}'.format(name, val)
			# check if val is not given and if required is true
			if isinstance(required, bool):
				if required is True and val is None:
					print "required argument not given: {}".format(name)
					sys.exit(2)				
			else:
				raise TypeError('get() keyword "required" expecting an input of type boolean')
			# default value must behave type/length declarations
			if arg_type in [list, tuple] and val != None:
				# assert that the list is of the specified length
				val = assert_list(val)		
			# val has been assigned its default and required has had a chance to exit
			return val

		# parse arg to a given type
		if arg_type:
			if arg_type not in self._acceptable_types:
				raise TypeError('Not an acceptable type: {}. Acceptable types: {}'.format(
							arg_type, self._acceptable_types))

			# must use get_type() for lists
			if arg_type in [list, tuple]:
				val = self.get_type(val, **kwargs)
				# a list of length 1 will be returned as an int/float
				if not isinstance(val, list):
					val = [val]
				# assert that the list is of the specified length
				val = assert_list(val)
				if arg_type == tuple:
					val = tuple(val)
				return val
			
			# attempt to convert to arg_type
			return change_type(val)
		else:
			return self.get_type(val, **kwargs)

	# check self._inputs and self._overrides for name and return its value if it exists
	def get(self, name, *args, **kwargs):
		# Option kwargs
		alt_names = kwargs.get('alt_name', [])
		alt_names = kwargs.get('alt_names', alt_names)
		# ensure alt_names is a list
		if not isinstance(alt_names, list):
			alt_names = [alt_names]
		help_str = kwargs.get('help')
		required = kwargs.get('required', False)
		hide = kwargs.get('hide', False)
		# Group kwargs
		group = kwargs.get('group')
		group = kwargs.get('g', group)
		mutually_exclusive = kwargs.get('mutually_exclusive', False)
		mutually_exclusive = kwargs.get('mutexc', mutually_exclusive)
		group_help_str = kwargs.get('group_help')
		group_help_str = kwargs.get('ghelp', group_help_str)

		# the only accepted arg is the default value for the current arg
		default = None
		if len(args) > 0:
			default = args[0]
			if len(args) > 1:
				errmsg = 'GetArgs.get(name, [default], [**kwargs]...) unexpected argument(s): {}'.format(args[1:])
				raise TypeError(errmsg)

		# do not allow 2 options to share the same name or alt_name
		for n in [name] + alt_names:
			for g in self.groups.values():
				if g._has_option(n):
					o = g._get_option(n)
					# ignore if n is the help option
					if o._is_arg('--help'): continue
					exception_str = "Option name '{}' has already been assigned to Option: '{}'".format(n, o.name)
					if o.help:
						exception_str = "{}: {}'".format(exception_str[:-1], o.help)
					raise ValueError(exception_str) 
		
		# parse the input and store its value to be given to a new Option object
		val = self.parse_arg(name, default, **kwargs)

		# determine if the current option has been given as an arg
		is_given = False
		if any(n in self._inputs for n in [name] + alt_names):
			is_given = True

		# add a new Option to the appropriate group
		if required is True:
			self.groups['Required'].append(Option(name, alt_names=alt_names, help=help_str, val=val, default=default, is_given=is_given, hide=hide))
		elif group:
			# determine if the group already exists
			if group in self.groups:
				self.groups[group].append(Option(name, alt_names=alt_names, help=help_str, val=val, default=default, is_given=is_given, hide=hide))
			# create a new group and append a new Option
			else:
				self.groups[group] = Group(group, help=group_help_str,
										mutually_exclusive=mutually_exclusive,
										option=Option(name, alt_names=alt_names, help=help_str, val=val, default=default, is_given=is_given, hide=hide))
		# no group specified, add to default group
		else:
			self.groups['Default'].append(Option(name, alt_names=alt_names, help=help_str, val=val, default=default, is_given=is_given, hide=hide))

		# determine if all inputs have been parsed
		self._num_parsed += 1
		if self._num_parsed == self.num_args:
			self.finish()
		return val	

	def is_given(self, arg_name):
		"""Determine if an ARG_NAME is given as an arg"""
		for group in self.groups.values():
			for arg in group:
				if arg._is_arg(arg_name) is True:
					return arg.is_given()
		# arg_name is not an arg that has been parsed
		return False

	def is_parsed(self, arg_name):
		"""Iterate through all parsed arguments and determine if arg has been parsed"""
		for group in self.groups.values():
			for arg in group:
				if arg._is_arg(arg_name) is True:
					return True
		# arg_name not found
		return False

	def get_parsed_args(self):
		"""Return a dictionary of args that were parsed through the get() function
		This does not include default values."""
		d = {}
		for group in self.groups.values():
			for arg in group:
				if arg.is_given():
					d[arg.name] = arg.get_val()
		return d

	def get_unparsed_args(self):
		"""Return a dictionary of args that have not been parsed through the get() function.
		This does not include default values."""
		d = {}
		for arg, val in self._inputs.items():
			if self.is_parsed(arg) is False:
				d[arg] = val
		return d

	def get_all_args(self):
		"""Return a dictionary of all arguments given to this GetArgs object"""
		d = self.get_parsed_args()
		# include unparsed args
		d.update(self.get_unparsed_args())
		return d		

	def get_all(self):
		"""Return a dictionary of all args including default values of args that are not given,
		and args that have not been defined by a call to get()."""
		d = {}
		for g in self.groups.values():
			d.update(g.get_inputs())
		# include unparsed args
		d.update(self.get_unparsed_args())
		return d	

	def get_posargs(self):
		return self._posargs

	def _get_not_parsed(self, asdict=True):
		"""Return a list of args that have not been parsed. Keep the order of the args as they were given
		and ignore any help args."""
		not_parsed = []
		for arg in self._arg_list:
			if arg in self._help_args: continue
			if self.is_parsed(arg) is False:
				not_parsed.append(arg)
		return not_parsed

	def create_arg_list(*argdict):
		"""Create a list that could be passed to a GetArgs object as an arg list.
		By default, the raw args that were passed to this object will be returned.
		If a dictionary (arg_name, arg_val) is given, attempt to make an arg_list from that dictionary.
		"""
		try:
			argdict = argdict[0]
		except IndexError:
			# if no argdict was given, return the raw inputs of this object
			return self._raw_inputs
		else:
			arg_list = []
			for arg, val in argdict.items():
				if isinstance(val, list) or isinstance(val, tuple):
					arg_list.append('{}={}'.format(arg, ','.join([str(v) for v in val])))
				elif val is True:
					arg_list.append(arg)
				else:
					arg_list.append('{}={}'.format(arg, val))
			return arg_list


	def get_options_help_str(self, use_module_name=False):
		# display the information of each group
		lines = []
		for group in self.groups.values():
			if len(group) == 0: continue
			if group.hide() is True: continue
			lines.append('')
			if group.name == 'Option':
				lines.append(group.get_help_str(show_defaults=self._show_defaults, alt_name=self._app_name.replace('.py','')))
			else:
				lines.append(group.get_help_str(show_defaults=self._show_defaults))
		help_str = '\n'.join(lines)
		while help_str.endswith('\n'):
			help_str = help_str[:-1]
		return help_str

	def get_help_str(self):
		help_str = 'Usage: {}'.format(self._app_name)
		for group in self.groups.values():
			help_str += ' {}'.format(group.usage())

		# include any posargs
		if self._posargs_names:
			for i, arg_name in enumerate(self._posargs_names):
				if i+1 > self._num_required_posargs:
					help_str += ' [{}]'.format(arg_name)
				else:
					help_str += ' {}'.format(arg_name)
			if self._num_posargs > len(self._posargs_names):
				help_str += '...'
		elif self._num_posargs:
			for i in range(self._num_posargs):
				if i+1 > self._num_required_posargs:
					help_str += ' [ARG{}]'.format(i+1)
				else:
					help_str += ' ARG{}'.format(i+1)

		# if self._num_posargs == 1:
		# 	help_str += ' {}'.format(self._posargs_names.upper())
		# elif self._num_posargs > 1:
		# 	help_str += ' {}...'.format(self._posargs_names.upper())

		if self._help:
			help_str += '\n{}'.format(self._help)
		help_str += '\n'
		return help_str	+ self.get_options_help_str()

	def display_help(self):
		print self.get_help_str()

	def finish(self, **kwargs):
		"""Check for any bad arguments the user may have given and return positional arguments
		as a tuple of (<required>, <optional>). If <required> is not given, then return only
		<optional>. If <all_required> is given, then return only a list of required positional
		arguments.
		Possible kwargs:
		  required: A string to display the name of a required positional argument for --help.
		  			If given, then the first positional argument will be required.
		  all_required: A string to display the name of a required positional argument for --help
		                when there can be more than one required positional argument. If given, then
		                return only a list of required arguments. <all_required> overrides <required>
		                and <optional>.
		  optional: A string to display the name of a optional positional arguments for --help.
		  			<optional> will be a list containing all positional arguments aside from 
		  			<required> if it was given.
		  type:  If given, attempt to parse positional arguments to <type>.
		  rtype: If given, attempt to parse the required argument to <rtype>, this overrides <type>.
		  otype: If given, attempt to parse the optional arguments to <otype>, this overrides <type>.
		  show_args: If True, output the command line arguments that were given. Default True.
		  ignore:  If True, do not exit if a bad arg is given
		"""
		ignore = kwargs.get('ignore', False)
		required = kwargs.get('required')
		all_required = kwargs.get('all_required')
		optional = kwargs.get('optional')
		arg_type = kwargs.get('type')
		required_type = kwargs.get('rtype', arg_type)
		optional_type = kwargs.get('otype', arg_type)
		show_args = kwargs.get('show_args', self._show_args)		

		posargs = []
		# a list of valid arguments from get() function
		valid = []
		for group in self.groups.values():
			for option in group:
				valid.append(option.name)
				valid.extend(option.alt_names)
		# determine if any args have not been parsed
		not_parsed = self._get_not_parsed()

		# find the last position of a valid arg
		last = 0
		for i, arg in enumerate(self._arg_list):
			if arg in valid:
				last = i

		# remove posargs from not_parsed
		for arg in not_parsed[:]:
			if self._arg_list.index(arg) >= last:
				posargs.append(arg)
				not_parsed.remove(arg)		

		if ignore is False:
			# warn the user that there are unrecognized args
			if len(not_parsed) > 0:
				for arg in not_parsed:
					# args that are not given with a value (i.e., flags) are assigned a boolean value of True if given
					if self._inputs[arg] != True:
						arg = self.get_raw_input(arg)
					print "{}: unrecognized arg '{}'".format(self._app_name, arg)
				print "Try '{} --help' for more information.".format(self._app_name)
				sys.exit()

		# handle help messages
		if self._help_min is True:
			print self._help
		if '--help-options-only' in self._inputs.keys() + self._overrides.keys():
			if self._help_exit is False: self.groups['Default']._get_option('--help').hide = True
			for group in self.groups.values():
				group.sort()
			print self.get_options_help_str(use_module_name=True)
			if self._help_exit is True: sys.exit()
		if any(h in self._inputs.keys() + self._overrides.keys() for h in self._help_args):
			if self._help_exit is False: self.groups['Default']._get_option('--help').hide = True
			# sort all groups
			for group in self.groups.values():
				group.sort()
			self.display_help()
			if self._help_chain is True:
				for arg in self._help_args:
					try: self._inputs.pop(arg)
					except KeyError: pass
				self._inputs['--help-options-only'] = True
			if self._help_exit is True: sys.exit()	

		# ensure the proper posargs are given
		if self._num_required_posargs:
			if len(posargs) < self._num_required_posargs:
				print "expecting '{}' positional arguments. Got only: {}".format(self._num_required_posargs, len(posargs))
				sys.exit(2)
		if self._num_posargs:
			if len(posargs) > self._num_posargs:
				print "found '{}' posargs. A maximum of '{}' can be given".format(len(posargs), self._num_posargs)
				sys.exit(2)	

		if show_args is True:
			show = sys.argv[:]
			show[0] = os.path.basename(show[0])
			print txt.header(' '.join(show))

		# attempt to convert posargs to arg_type
		if arg_type:
			for i in range(0, len(posargs)):
				try:
					posargs[i] = arg_type(posargs[i])
				except ValueError:
					print "could not parse posarg '{}' to type: {}".format(posargs[i], arg_type)
					sys.exit(2)

		return posargs


############### TO-DO ################
# - add built-in args: --__setdefaults__ FILE, --__savedefaults__ FILE, --__cleardefaults__ 

############### BUGS #################




