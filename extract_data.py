import json
import gzip
import sys

input_path  = sys.argv[1]
output_path = sys.argv[2]
quantity		= int(sys.argv[3])

i = 0
f = open(output_path, 'w')
g = gzip.open(input_path, 'r')
for l in g:
	line = json.dumps(eval(l))
	f.write(line + '\n')
	i += 1
	if i == quantity:
		f.close()
		exit()

