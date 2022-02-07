import sys, getopt

def main(argv):
   try:
        opts, args = getopt.getopt(argv,"hi:o:",["ifile=","ofile="])
   except getopt.GetoptError:
        # ERROR
        sys.exit(2)
   for opt, arg in opts:
        if opt == '-h':
            # Help
            sys.exit()
        elif opt in ("-i"):
            inputfile = arg
        elif opt in ("-o",):
            outputfile = arg

