AGE_BOUNDARIES=[-0.1,0,0.6,1.5,2,3]

GENDER_BOUNDARIES=[0,0.6]

COUNTRY_BOUNDARIES=[-0.5,-0.3,-0.15,0,0.22,0.26,1]

ETHNICITY_BOUNDARIES=[-1,-0.4,-0.25,0,0.118,0.5,2]

EDUCATION_BOUNDARIES=[-2,-1.5,-1.3,-1,-0.5,0,1,1.5,2]

USER = [
	"CL4", "CL5", "CL6"
]

NON_USER = [
	"CL0", "CL1", "CL2", "CL3"
]

MAPPED_CODES = {k:v for v, k in enumerate(USER + NON_USER)}

CSV_COLUMNS = [
	"age", "gender", "education", "country", "ethnicity",
	"nscore", "escore", "oscore", "ascore", "cscore",
	"impulsive", "ss", "alcohol", "amphet", "amyl",
	"benzos", "caff", "cannabis", "choc", "coke", "crack",
	"ecstasy", "heroin", "ketamine", "legalh", "LSD",
	"meth", "mushrooms", "nicotine", "semer", "VSA"
]

FEATURE_COLUMNS = [
	"age", "gender", "education", "country", "ethnicity",
	"nscore", "escore", "oscore", "ascore", "cscore",
	"impulsive", "ss"
]

def line_diff(first, second):
	return [x for x in first if x not in second]

USAGE_COLUMNS = line_diff(CSV_COLUMNS, FEATURE_COLUMNS)

NUMBERED_COLUMNS = {k:v for v, k in enumerate(CSV_COLUMNS)}

TARGET = ["alcohol"]

MODEL_DIR = '/tmp/drug_model'