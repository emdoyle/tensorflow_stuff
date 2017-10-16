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

def add_image_dir(line):
	index = line.find(INDICATOR)
	if index != -1:
		split_idx = index+len(INDICATOR)
		line = line[:split_idx] + IMAGE_DIR + line[split_idx:]
	return line

def swap_caption_to_png(line):
	index = line.find(CAPTION_INDICATOR)
	if index != -1:
		split_idx = index
		offset = len(CAPTION_INDICATOR+CAPTION_DIR)
		line = line[:split_idx] + INDICATOR + IMAGE_DIR + line[(split_idx+offset):]
	return line

def remove_image_dirs(line):
	while line.find(IMAGE_DIR) != -1:
		index = line.find(IMAGE_DIR)
		line = line[:index] + line[index+len(IMAGE_DIR):]
	return line

with open(FILENAME, 'r') as f:
	line = f.readline()
	while line:
		if line.find(IMAGE_DIR) == -1:
			if line.find(INDICATOR) != -1:
				line = add_image_dir(line)
			elif line.find(CAPTION_INDICATOR) != -1:
				line = swap_caption_to_png(line)
		else:
			line = remove_image_dirs(line)
			line = add_image_dir(line)
		temp.append(line)
		line = f.readline()

with open(FILENAME, 'w') as f:
	f.writelines(temp)