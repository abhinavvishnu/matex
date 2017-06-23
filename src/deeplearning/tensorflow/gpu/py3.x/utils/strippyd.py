import re
import sys

ver = sys.version
ver_list = ver.split()
ver_list0 = ver_list[0]
verSplit = ver_list0.split('.')
print('{}.{}'.format(verSplit[0],verSplit[1]))

