import time

start_time = time.time()

def log(message='', end='\n'):
	print('[%4d] %s' % (time.time() - start_time, message), end=end)
