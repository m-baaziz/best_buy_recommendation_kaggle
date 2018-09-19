import re
import time
from datetime import datetime

def timestring_to_ms(timestring):
	try:
		dt = datetime.strptime(timestring, '%Y-%m-%d %H:%M:%S.%f')
	except Exception as e:
		try:
			dt = datetime.strptime(timestring, '%Y-%m-%d %H:%M:%S')
		except Exception as e:
			dt = datetime.strptime(timestring, '%Y-%m-%d')
	return time.mktime(dt.timetuple()) * 1000

def dict_list_mean_by_key(d, key):
	return sum([v[key] for k,v in d.iteritems()]) / len(d)

def dict_list_mode_by_key(d, key):
	items = [v[key] for k,v in d.iteritems()]
	return max(items, key=items.count)	

def keep_digits(s):
	return ''.join(re.findall('[0-9\\s]', s))

def keep_letters(s):
	return ''.join(re.findall('[a-zA-Z\\s]', s))

def remove_from_string(s, pattern):
	return re.sub(pattern, '', s)