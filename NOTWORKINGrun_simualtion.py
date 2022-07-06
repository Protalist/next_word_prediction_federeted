from multiprocessing.dummy import Process
import sys
from new_server import server
from new_client_random import client

import time

def main(num_client):
  proc = []
  s = Process(target=server,args=(num_client,))
  s.start()
  proc.append(s)
  time.sleep(10)
  for i in  range(num_client):
    time.sleep(2)
    p = Process(target=client, args=(num_client, str(i)))
    p.start()
    proc.append(p)
  for p in proc:
    p.join()

if __name__ == "__main__":
  b=0
  try:
      b = int(sys.argv[1])
  except IndexError:
      b = 10
  main(b)