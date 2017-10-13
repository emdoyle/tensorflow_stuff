import sys

if len(sys.argv) != 2:
	print("Usage: fix_image_paths.py [filename]")
	sys.exit(2)

INDICATOR = "![png]("
IMAGE_DIR = "assets/"

FILENAME = sys.argv[1]
temp = []

with open(FILENAME, 'r') as f:
	line = f.readline()
	while line:
		index = line.find(INDICATOR)
		if index != -1:
			split_idx = index+len(INDICATOR)
			line = line[:split_idx] + IMAGE_DIR + line[split_idx:]
		temp.append(line)
		line = f.readline()

with open(FILENAME, 'w') as f:
	f.writelines(temp)