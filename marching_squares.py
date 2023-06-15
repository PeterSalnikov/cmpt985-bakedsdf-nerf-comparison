#lab3 of 464
import numpy as np

def trace(phi, e1, e2):
  """
  Args:
  phi: The SDF grid, shape: (N,N)
  e1: The first edge point, shape: (2)
  e2: The second edge point, shape: (2)
  Returns:
  p: The intersection point, shape: (2)
  check: Whether there is an intersection, shape: (1) bool
  """
  p = np.array([np.nan, np.nan])

  sdf1 = phi[e1[0], e1[1]]
  sdf2 = phi[e2[0], e2[1]]

  if np.sign(sdf1)*np.sign(sdf2) < 0:
    e2e1 = (e2-e1)
    p = e1 - sdf1/(sdf2-sdf1) * e2e1
    return p, True
  return p, False

def marching_squares(phi, SKIP=6):

  edges = []
  for x in range(SKIP, phi.shape[0]-SKIP, SKIP):
    for y in range(SKIP, phi.shape[1]-SKIP, SKIP):
      grd = np.zeros((4, 2), dtype=int)
      grd[0] = np.array([x, y])
      grd[1] = np.array([x + SKIP, y])
      grd[2] = np.array([x + SKIP, y + SKIP])
      grd[3] = np.array([x, y + SKIP])
      p0, b0 = trace(phi, grd[0], grd[1])
      p1, b1 = trace(phi, grd[1], grd[2])
      p2, b2 = trace(phi, grd[2], grd[3])
      p3, b3 = trace(phi, grd[3], grd[0])
      P = np.array([p0, p1, p2, p3])
      B = np.array([b0, b1, b2, b3])
      E = P[B==True, :]
      if E.shape[0] == 2: # no handling of degenerate cases, to see more, please check http://users.polytech.unice.fr/~lingrand/MarchingCubes/algo.html
        edges.append(E)

  return edges
