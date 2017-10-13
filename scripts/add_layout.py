import sys

if len(sys.argv) != 2:
	print("Usage: add_layout.py [filename]")
	sys.exit(2)

LAYOUT = "---\nlayout: notebook\n---\n"

FILENAME = sys.argv[1]
temp = []

with open(FILENAME, 'r') as f:
	temp = f.read()

with open(FILENAME, 'w') as f:
	f.write(LAYOUT)
	f.write(temp)