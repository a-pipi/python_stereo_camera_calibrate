import cv2 as cv
import glob
import numpy as np
import matplotlib.pyplot as plt

rows = 6 #number of checkerboard rows.
columns = 9 #number of checkerboard columns.
world_scaling = 13 #change this to the real world square size. Or not.
_show = True

def calibrate_camera(images_folder):
    """
    Calibrates a camera to get its internal parameters. It uses a checkerboard
    in the frames to determine the parameters. It shows the frames with the 
    found checkerboard. 
    
    input:
    - images folder (str)

    output:
    - Camera matrix (mtx)
    - Camera distortion coefficients (dist)
    """
    # sort images, read them and put them in a list
    images_names = sorted(glob.glob(images_folder))
    # images = []
    # for imname in images_names:
    #     im = cv.imread(imname, 1)
    #     images.append(im)

    images = [cv.imread(imname, 1) for imname in images_names]
  
    #criteria used by checkerboard pattern detector.
    #Change this if the code can't find the checkerboard
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
 
    #coordinates of squares in the checkerboard world space
    objp = np.zeros((rows*columns,3), np.float32)
    objp[:,:2] = np.mgrid[0:rows,0:columns].T.reshape(-1,2)
    objp = world_scaling* objp
 
    #frame dimensions. Frames should be the same size.
    width = images[0].shape[1]
    height = images[0].shape[0]

    # setup focal length and pixelsize
    focal_length = 16 # in mm
    pixel_size = 2.4e-3 # in mm

    # get camera parameters
    fx = focal_length/pixel_size
    fy = focal_length/pixel_size
    cx = width/2
    cy = height/2

    intrinsic_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
 
    #Pixel coordinates of checkerboards
    imgpoints = [] # 2d points in image plane.
 
    #coordinates of the checkerboard in checkerboard world space.
    objpoints = [] # 3d point in real world space
    idx = 0
    for frame in images:
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        #find the checkerboard
        ret, corners = cv.findChessboardCorners(gray, (rows, columns), None)
        cv.destroyAllWindows()

        
        if ret == True:

            #Convolution size used to improve corner detection. Don't make this too large.
            conv_size = (11, 11)

            #opencv can attempt to improve the checkerboard coordinates
            corners = cv.cornerSubPix(gray, corners, conv_size, (-1, -1), criteria)
            if _show:
                cv.drawChessboardCorners(frame, (rows,columns), corners, ret)
                res_frame = cv.resize(frame, (1080,720))
                cv.imshow(images_names[idx], res_frame)
                k = cv.waitKey(100)
                print(images_names[idx])

            objpoints.append(objp)
            imgpoints.append(corners)

            idx += 1

 
    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, (width, height), None, None)#intrinsic_matrix, None, flags=cv.CALIB_USE_INTRINSIC_GUESS)
    print('rmse:', ret)
    # print('camera matrix:\n', mtx)
    # print('distortion coeffs:', dist)
    # print('Rs:\n', rvecs)
    # print('Ts:\n', tvecs)
 
    return mtx, dist

def stereo_calibrate(mtx1, dist1, mtx2, dist2, frames_1, frames_2):
    """
    Stereo calibrates two camera to get the rotation and translation between camera 1 and camera 2.

    input:
    - Camera matrix 1 (mtx1)
    - Camera matrix 2 (mtx2)
    - Distortion coefficient camera 1 (dist1)
    - Distortion coefficient camera 2 (dist2)
    - Frames of camera 1 (frames_1)
    - Frames of camera 2 (frames_2)

    output:
    - Rotation matrix between camera 1 and camera 2 (R)
    - Translation vector between camera 1 and camera 2 (T)

    """
    #read the synched frames
    c1_images_names = sorted(glob.glob(frames_1))
    c2_images_names = sorted(glob.glob(frames_2))

    c1_images = []
    c2_images = []
    for im1, im2 in zip(c1_images_names, c2_images_names):
        _im = cv.imread(im1, 1)
        c1_images.append(_im)
 
        _im = cv.imread(im2, 1)
        c2_images.append(_im)
 
    #change this if stereo calibration not good.
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.0001)
 
    #coordinates of squares in the checkerboard world space
    objp = np.zeros((rows*columns,3), np.float32)
    objp[:,:2] = np.mgrid[0:rows,0:columns].T.reshape(-1,2)
    objp = world_scaling* objp
 
    #frame dimensions. Frames should be the same size.
    width = c1_images[0].shape[1]
    height = c1_images[0].shape[0]
 
    #Pixel coordinates of checkerboards
    imgpoints_left = [] # 2d points in image plane.
    imgpoints_right = []
 
    #coordinates of the checkerboard in checkerboard world space.
    objpoints = [] # 3d point in real world space
    
    idx = 0
    for frame1, frame2 in zip(c1_images, c2_images):
        # threshold iamges
        gray1 = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
        gray2 = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)

        # find corners of checkerboards
        c_ret1, corners1 = cv.findChessboardCorners(gray1, (rows, columns), None)
        c_ret2, corners2 = cv.findChessboardCorners(gray2, (rows, columns), None)
 
        if c_ret1 == True and c_ret2 == True:
            # get subpixel coordinates of corners
            corners1 = cv.cornerSubPix(gray1, corners1, (11, 11), (-1, -1), criteria)
            corners2 = cv.cornerSubPix(gray2, corners2, (11, 11), (-1, -1), criteria)
  
            if idx == 0:
                corner_point = [corners1[0], corners2[1]]

            if _show:
                # draw checkerboard on images and show images
                cv.drawChessboardCorners(frame1, (rows, columns), corners1, c_ret1)
                res_frame = cv.resize(frame1, (1080,720))
                cv.imshow(c1_images_names[idx], res_frame)
    
                cv.drawChessboardCorners(frame2, (rows, columns), corners2, c_ret2)
                res_frame = cv.resize(frame2, (1080,720))
                cv.imshow(c2_images_names[idx], res_frame)
                k = cv.waitKey(100)
                cv.destroyAllWindows()
            
            # append object and image points to variables
            objpoints.append(objp)
            imgpoints_left.append(corners1)
            imgpoints_right.append(corners2)
            idx += 1
    
    # perform stereo calibration
    stereocalibration_flags = cv.CALIB_RATIONAL_MODEL+cv.CALIB_FIX_INTRINSIC+cv.CALIB_FIX_PRINCIPAL_POINT
    ret, CM1, dist1, CM2, dist2, R, T, E, F = cv.stereoCalibrate(objpoints, imgpoints_left, imgpoints_right, mtx1, dist1,
                                                                 mtx2, dist2, (width, height), criteria = criteria, flags = stereocalibration_flags)
    # print("Corner points: ", corner_point)
    print("RMSE: ", ret)
    return R, T


def triangulate(mtx1, mtx2, R, T, trian_corners):
    
    uvs1 = trian_corners[0]
 
    uvs2 = trian_corners[1]
 
    uvs1 = np.array(uvs1)
    uvs2 = np.array(uvs2)
 
 
    frame1 = cv.imread('images/synced/stereoLeft/imageL0.png')
    frame2 = cv.imread('images/synced/stereoRight/imageR0.png')
 
    plt.imshow(frame1[:,:,[2,1,0]])
    plt.scatter(uvs1[:,0], uvs1[:,1])
    plt.show() #this call will cause a crash if you use cv.imshow() above. Comment out cv.imshow() to see this.
 
    plt.imshow(frame2[:,:,[2,1,0]])
    plt.scatter(uvs2[:,0], uvs2[:,1])
    plt.show()#this call will cause a crash if you use cv.imshow() above. Comment out cv.imshow() to see this
 
    #RT matrix for C1 is identity.
    RT1 = np.concatenate([np.eye(3), [[0],[0],[0]]], axis = -1)
    P1 = mtx1 @ RT1 #projection matrix for C1
 
    #RT matrix for C2 is the R and T obtained from stereo calibration.
    RT2 = np.concatenate([R, T], axis = -1)
    P2 = mtx2 @ RT2 #projection matrix for C2
 
    def DLT(P1, P2, point1, point2):
 
        A = [point1[1]*P1[2,:] - P1[1,:],
             P1[0,:] - point1[0]*P1[2,:],
             point2[1]*P2[2,:] - P2[1,:],
             P2[0,:] - point2[0]*P2[2,:]
            ]
        A = np.array(A).reshape((4,4))
        #print('A: ')
        #print(A)
 
        B = A.transpose() @ A
        from scipy import linalg
        U, s, Vh = linalg.svd(B, full_matrices = False)
 
        # print('Triangulated point: ')
        # print(Vh[3,0:3]/Vh[3,3])
        return Vh[3,0:3]/Vh[3,3]
 
    p3ds = []
    for uv1, uv2 in zip(uvs1, uvs2):
        _p3d = DLT(P1, P2, uv1, uv2)
        p3ds.append(_p3d)
    p3ds = np.array(p3ds)
 
    from mpl_toolkits.mplot3d import Axes3D
 
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.set_xlim3d(-15, 5)
    # ax.set_ylim3d(-10, 10)
    # ax.set_zlim3d(10, 30)
    print(p3ds)
    # ax.plot(xs = p3ds[0][0], ys = p3ds[0][1], zs = p3ds[0][2])
    # connections = [[0,1], [1,2], [2,3], [3,4], [1,5], [5,6], [6,7], [1,8], [1,9], [2,8], [5,9], [8,9], [0, 10], [0, 11]]
    # for _c in connections:
    #     # print(p3ds[_c[0]])
    #     # print(p3ds[_c[1]])
    #     ax.plot(xs = [p3ds[_c[0],0], p3ds[_c[1],0]], ys = [p3ds[_c[0],1], p3ds[_c[1],1]], zs = [p3ds[_c[0],2], p3ds[_c[1],2]], c = 'red')
    # ax.set_title('This figure can be rotated.')
    #uncomment to see the triangulated pose. This may cause a crash if youre also using cv.imshow() above.
    plt.show()

if __name__ == "__main__":
    # calibrate both cameras
    mtx1, dist1 = calibrate_camera(images_folder = 'images/synced/camera1/*')
    mtx2, dist2 = calibrate_camera(images_folder = 'images/synced/camera2/*')
    cv.destroyAllWindows()

    # stereo calibrate cameras
    R, T = stereo_calibrate(mtx2, dist2, mtx1, dist1, 'images/synced/camera1/*', 'images/synced/camera2/*')
    cv.destroyAllWindows()


    print(np.linalg.norm(T))
    M = np.empty((4,4))
    M[:3,:3] = R
    M[:3,3] = T.T[0]
    M[3,:] = [0,0,0,1]

    print(np.dot(M, [0,0,0,1]))

    # Save parameters to XML file
    cv_file = cv.FileStorage('stereoCalibration.XML', cv.FileStorage_WRITE)
    cv_file.write("mtx1", mtx1)
    cv_file.write("mtx2", mtx2)
    cv_file.write("dist1", dist1)
    cv_file.write("dist2", dist2)
    cv_file.write("R", R)
    cv_file.write("T", T)
