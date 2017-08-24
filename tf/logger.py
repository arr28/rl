import datetime

def log(message, end='\n'):
  print('%s %s' % (datetime.datetime.now(), message), end=end, flush=True)
  
def log_progress():
  print('.', end='', flush=True)