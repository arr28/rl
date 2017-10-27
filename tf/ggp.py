import breakthrough as bt
import little_golem as lg
import numpy as np
import re

from http.server import HTTPServer, BaseHTTPRequestHandler
from policy import CNPolicy

PRIMARY_CHECKPOINT = 'model.epoch99.hdf5'

state = None
role = None
policy = None

def run_ggp():
  global policy
  policy = CNPolicy(checkpoint=PRIMARY_CHECKPOINT)
  httpd = HTTPServer(('', 9147), GGPRequestHandler)
  httpd.serve_forever()
  # The above method never returns.  Don't add anything below this line.
    
class GGPRequestHandler(BaseHTTPRequestHandler):
  def do_GET(self):
    print('GET of %s' % self.path)
    self.send_response(200)
    self.end_headers()

  def do_POST(self):
    global state
    global role
    global policy
    
    print('POST to %s' % self.path)
    length = self.headers.get('content-length')
    request = self.rfile.read(int(length)).decode('ascii')
    print('Request was %s' % request)

    response = ''    
    if request.startswith('( INFO )'):
      response = '( (name Mimic) (status available) )'
    elif request.startswith('( START '):
      state = bt.Breakthrough()
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
      if move != 'NIL':
        print('Raw move was %s' % move)
        move = list(move)
        src_col = 8 - int(move[0])
        src_row = int(move[1]) - 1
        dst_col = 8 - int(move[2])
        dst_row = int(move[3]) - 1
        move = (src_row, src_col, dst_row, dst_col)
        print('Move was %s' % lg.encode_move(move))
        state.apply(move)
      else:
        print('First move')

      if state.player == role:
        prediction = policy.get_action_probs(state)
        index = np.argsort(prediction)[::-1][0]
        move = bt.convert_index_to_move(index, state.player)
        print('Playing %s' % (lg.encode_move(move)))
        src_row = move[0] + 1
        src_col = 8 - move[1]
        dst_row = move[2] + 1
        dst_col = 8 - move[3]
        response = '( move %d %d %d %d )' % (src_col, src_row, dst_col, dst_row)
      else:
        print('Not our turn - no-op')
        response = 'noop'

    print('Responding with %s' % response)    
    response_bytes = response.encode('ascii')
    self.send_response(200)
    self.send_header('Content-Length', len(response_bytes))
    self.end_headers()
    self.wfile.write(response_bytes)
    self.wfile.flush()

if __name__ == "__main__":
  run_ggp()
    