import sys

if len(sys.argv) != 2:
	print("Usage: fix_image_paths.py [filename]")
	sys.exit(2)

INDICATOR = "![png]("
IMAGE_DIR = "assets/"
CAPTION_DIR = "files/"
CAPTION_INDICATOR = "![caption]("

FILENAME = sys.argv[1]
temp = []

with open(FILENAME, 'r') as f:
	line = f.readline()
	while line:
		if line.find(IMAGE_DIR) == -1:
			index = line.find(INDICATOR)
			if index != -1:
				split_idx = index+len(INDICATOR)
				line = line[:split_idx] + IMAGE_DIR + line[split_idx:]
			index = line.find(CAPTION_INDICATOR)
			if index != -1:
				split_idx = index
				offset = len(CAPTION_INDICATOR+CAPTION_DIR)
				line = line[:split_idx] + INDICATOR + IMAGE_DIR + line[(split_idx+offset):]
		temp.append(line)
		line = f.readline()

with open(FILENAME, 'w') as f:
	f.writelines(temp)