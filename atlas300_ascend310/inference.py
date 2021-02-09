import numpy as np

f=open("output.txt",'r')
content=f.readlines()
f.close()
print(content[0].split(" ")[:90])

