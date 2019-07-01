import math
import numpy as np

def dotproduct(v1, v2):
  return sum((a*b) for a, b in zip(v1, v2))

def length(v):
  return math.sqrt(dotproduct(v, v))

def angle(v1, v2):
  return math.acos(dotproduct(v1, v2) / (length(v1) * length(v2)))

def getProjections((originX, originY), (tailX, tailY), (a, b)):
    arrow = (a,b)
    vector = (tailX - originX, tailY - originY)
    
    l = length(vector)
    coef = dotproduct(arrow,vector) / (l * l) 

    
    parallel = (coef * vector[0], coef * vector[1])
    perp = (arrow[0] - parallel[0], arrow[1] - parallel[1])
    
    return (parallel, perp)

