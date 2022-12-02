import cv2
import numpy as np
from scipy import linalg


def triangulate(mtx1, mtx2, R, T, pnt1, pnt2):
  #RT matrix for C1 is identity.
  RT1 = np.concatenate([np.eye(3), [[0],[0],[0]]], axis = -1)
  P1 = mtx1 @ RT1 #projection matrix of camera 1

  #RT matrix for C2 is the R and T obtained from stereo calibration.
  RT2 = np.concatenate([R, T], axis = -1)
  P2 = mtx2 @ RT2 #projection matrix of camera 2

  # calculate 3d positions
  p3ds = []
  for uv1, uv2 in zip(pnt1, pnt2):
      _p3d = DLT(P1, P2, uv1, uv2)
      p3ds.append(_p3d)
  p3ds = np.array(p3ds)

  return p3ds





def DLT(P1, P2, point1, point2):
  A = [point1[1]*P1[2,:] - P1[1,:],
        P1[0,:] - point1[0]*P1[2,:],
        point2[1]*P2[2,:] - P2[1,:],
        P2[0,:] - point2[0]*P2[2,:]
      ]
  A = np.array(A).reshape((4,4))
  
  B = A.transpose() @ A
  U, s, Vh = linalg.svd(B, full_matrices = False)

  return Vh[3,0:3]/Vh[3,3]