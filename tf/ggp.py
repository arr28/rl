import breakthrough as bt
import re

from http.server import HTTPServer, BaseHTTPRequestHandler

match = None
role = None

def __static_init():
  global match
  global role

def run_ggp():
  httpd = HTTPServer(('', 9147), GGPRequestHandler)
  httpd.serve_forever()
  
class GGPRequestHandler(BaseHTTPRequestHandler):
  def do_GET(self):
    print('GET of %s' % self.path)
    self.send_response(200)
    self.end_headers()

  def do_POST(self):
    global match
    global role
    
    print('POST to %s' % self.path)
    length = self.headers.get('content-length')
    request = self.rfile.read(int(length)).decode('ascii')
    print('Request was %s' % request)

    response = ''    
    if request.startswith('( INFO )'):
      response = '( (name Mimic) (status available) )'
    elif request.startswith('( START '):
      match = bt.Breakthrough()
      if request.split(sep=' ')[3] == 'white':
        print('We are white')
        role = 0
      else:
        print('We are black')
        role = 1
      response = 'ready'
    elif request.startswith('( PLAY '):
      parsed_play_req = re.match(r'\( PLAY [^ ]* (.*)', request)
      # !! ARR Parse out the last move (remembering it might be NIL)
      move = parsed_play_req.group(1)
      move = move.replace('noop', '').replace('move', '').replace('(', '').replace(')', '').replace(' ', '')
      print('Move was %s' % move)
      if move != 'NIL':
        move = list(move)
        src_col = 8 - int(move[0])
        src_row = int(move[1]) - 1
        dst_col = 8 - int(move[2])
        dst_row = int(move[3]) - 1
        match.apply((src_row, src_col, dst_row, dst_col))

    print('Responding with %s' % response)    
    response_bytes = response.encode('ascii')
    self.send_response(200)
    self.send_header('Content-Length', len(response_bytes))
    self.end_headers()
    self.wfile.write(response_bytes)
    self.wfile.flush()

if __name__ == "__main__":
  run_ggp()
    