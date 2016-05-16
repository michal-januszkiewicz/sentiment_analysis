import json
import gzip
import sys

input_path  = sys.argv[1]
output_path = sys.argv[2]
quantity		= int(sys.argv[3])

i = 0
f = gzip.open(output_path + '.gz', 'w')
g = gzip.open(input_path, 'r')

for l in g:
	line = json.dumps(eval(l))
	f.write(bytes(line + '\n', 'UTF-8'))
	i += 1
	if i == quantity:
		f.close()
		exit()

f.close()

