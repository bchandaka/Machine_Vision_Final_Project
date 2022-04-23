import cv2
import datetime
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from sklearn.datasets import make_multilabel_classification

class Dataset_Handler():
    def __init__(self, sequence, lidar=True):
        
        # This will tell our odometry function if handler contains lidar info
        self.lidar = lidar
        
        # Set file paths and get ground truth poses
        self.seq_dir = './dataset/sequences/{}/'.format(sequence)
        self.poses_dir = './dataset/poses/{}.txt'.format(sequence)
        poses = pd.read_csv(self.poses_dir, delimiter=' ', header=None)
        
        # Get names of files to iterate through
        self.left_image_files = sorted(os.listdir(self.seq_dir + 'image_0'))
        self.right_image_files = sorted(os.listdir(self.seq_dir + 'image_1'))
        self.velodyne_files = sorted(os.listdir(self.seq_dir + 'velodyne'))
        self.num_frames = len(self.left_image_files)
        self.lidar_path = self.seq_dir + 'velodyne/'
        
        # Get calibration details for scene
        # P0 and P1 are Grayscale cams, P2 and P3 are RGB cams
        calib = pd.read_csv(self.seq_dir + 'calib.txt', delimiter=' ', header=None, index_col=0)
        self.P0 = np.array(calib.loc['P0:']).reshape((3,4))
        self.P1 = np.array(calib.loc['P1:']).reshape((3,4))
        self.P2 = np.array(calib.loc['P2:']).reshape((3,4))
        self.P3 = np.array(calib.loc['P3:']).reshape((3,4))
        # This is the transformation matrix for LIDAR
        self.Tr = np.array(calib.loc['Tr:']).reshape((3,4))
        
        # Get times and ground truth poses
        self.times = np.array(pd.read_csv(self.seq_dir + 'times.txt', 
                                          delimiter=' ', 
                                          header=None))
        self.gt = np.zeros((len(poses), 3, 4))
        for i in range(len(poses)):
            self.gt[i] = np.array(poses.iloc[i]).reshape((3, 4))
        
       
        # Will use generators to provide data sequentially to save RAM
        # Use class method to set up generators
        self.reset_frames()
        # Store original frame to memory for testing functions
        self.first_image_left = cv2.imread(self.seq_dir + 'image_0/' 
                                           + self.left_image_files[0], 0)
        self.first_image_right = cv2.imread(self.seq_dir + 'image_1/' 
                                           + self.right_image_files[0], 0)
        self.second_image_left = cv2.imread(self.seq_dir + 'image_0/' 
                                           + self.left_image_files[1], 0)
        if self.lidar:
            self.first_pointcloud = np.fromfile(self.lidar_path + self.velodyne_files[0],
                                                dtype=np.float32, 
                                                count=-1).reshape((-1, 4))
        self.imheight = self.first_image_left.shape[0]
        self.imwidth = self.first_image_left.shape[1]
            
            
    def reset_frames(self):
        # Resets all generators to the first frame of the sequence
        self.images_left = (cv2.imread(self.seq_dir + 'image_0/' + name_left, 0)
                            for name_left in self.left_image_files)
        self.images_right = (cv2.imread(self.seq_dir + 'image_1/' + name_right, 0)
                            for name_right in self.right_image_files)
        if self.lidar:
            self.pointclouds = (np.fromfile(self.lidar_path + velodyne_file, 
                                            dtype=np.float32, 
                                            count=-1).reshape((-1, 4))
                                for velodyne_file in self.velodyne_files)
        pass

def compute_left_disparity_map(img_left, img_right, matcher='bm', rgb=False, verbose=False):
    
    sad_window = 6
    num_disparities = sad_window * 16
    block_size = 11
    matcher_name = matcher
    
    if matcher_name == 'bm':
        matcher = cv2.StereoBM_create(numDisparities=num_disparities,
                                      blockSize=block_size)
        
    elif matcher_name == 'sgbm':
        matcher = cv2.StereoSGBM_create(numDisparities=num_disparities,
                                        minDisparity=0,
                                        blockSize=block_size,
                                        P1 = 8 * 1 * block_size ** 2,
                                        P2 = 32 * 1 * block_size ** 2,
                                        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
    
    if rgb:
        img_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
        img_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)
        
    start = datetime.datetime.now()
    disp_left = matcher.compute(img_left, img_right).astype(np.float32)/16
    end = datetime.datetime.now()
    
    if verbose:
        print(f'Time to compute disparity map using Stereo{matcher_name.upper()}', end-start)
        
    return disp_left

def decompose_projection_matrix(p):
    k, r, t, _, _, _, _ = cv2.decomposeProjectionMatrix(p)
    t = (t / t[3])[:3]
    
    return k, r, t

def calc_depth_map(disp_left, k_left, t_left, t_right, rectified=True):
    
    if rectified:
        b = t_right[0] - t_left[0]
    else:
        b = t_left[0] - t_right[0]
        
    f = k_left[0][0]
    
    disp_left[disp_left == 0.0] = 0.1
    disp_left[disp_left == -1.0] = 0.1
    
    depth_map = np.ones(disp_left.shape)
    depth_map = f * b / disp_left
    
    return depth_map
def stereo_2_depth(img_left, img_right, P0, P1, matcher='bm', rgb=False, verbose=False,
                   rectified=True):
    # Compute disparity map
    disp = compute_left_disparity_map(img_left,
                                      img_right,
                                      matcher=matcher,
                                      rgb=rgb,
                                      verbose=verbose)
    # Decompose projection matrices
    k_left, r_left, t_left = decompose_projection_matrix(P0)
    k_right, r_right, t_right = decompose_projection_matrix(P1)
    
    # Calculate depth map for left camera
    depth = calc_depth_map(disp, k_left, t_left, t_right)
    
    return depth
def pointcloud2image(pointcloud, imheight, imwidth, Tr, P0):
    
    pointcloud = pointcloud[pointcloud[:, 0] > 0]
    reflectance = pointcloud[:, 3]
    # Make pointcloud homogeneous (X, Y, Z, 1)
    pointcloud = np.hstack([pointcloud[:, :3], np.ones(pointcloud.shape[0]).reshape((-1,1))])
    
    # Transform points into 3D coordinate frame of camera
    cam_xyz = Tr.dot(pointcloud.T)
    # Clip off negative z values
    cam_xyz = cam_xyz[:, cam_xyz[2] > 0]
    
    depth = cam_xyz[2].copy()
    
    cam_xyz /= cam_xyz[2]    
    cam_xyz = np.vstack([cam_xyz, np.ones(cam_xyz.shape[1])])
    projection = P0.dot(cam_xyz)
    pixel_coordinates = np.round(projection.T, 0)[:, :2].astype('int')
    
    indices = np.where((pixel_coordinates[:, 0] < imwidth)
                       & (pixel_coordinates[:, 0] >= 0)
                       & (pixel_coordinates[:, 1] < imheight)
                       & (pixel_coordinates[:, 1] >= 0))
    
    pixel_coordinates = pixel_coordinates[indices]
    depth = depth[indices]
    reflectance = reflectance[indices]
    
    render = np.zeros((imheight, imwidth))
    for j, (u, v) in enumerate(pixel_coordinates):
        if u >= imwidth or u < 0:
            continue
        if v >= imheight or v < 0:
            continue
        render[v, u] = depth[j]
        
    return render

def extract_features(image, detector='sift', mask=None):

    if detector == 'sift':
        det = cv2.SIFT_create()
    elif detector == 'orb':
        det = cv2.ORB_create()
        
    kp, des = det.detectAndCompute(image, mask)

    
    return kp, des

def match_features(des1, des2, matching='BF', detector='sift', sort=False, k=2):
    
    if matching == 'BF':
        if detector == 'sift':
            matcher = cv2.BFMatcher_create(cv2.NORM_L2, crossCheck=False)
        elif detector == 'orb':
            matcher = cv2.BFMatcher_create(cv2.NORM_HAMMING2, crossCheck=False)
    elif matching == 'FLANN':
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        matcher = cv2.FlannBasedMatcher(index_params, search_params)
    
    matches = matcher.knnMatch(des1, des2, k=k)
        
    if sort:
        matches = sorted(matches, key=lambda x: x[0].distance)
        
    return matches

def visualize_matches(image1, kp1, image2, kp2, match):
    image_matches = cv2.drawMatches(image1, kp1, image2, kp2, match, None, flags=2)
    plt.figure(figsize=(16, 6), dpi=100)
    plt.imshow(image_matches)
    plt.show()

def filter_matches_distance(matches, dist_threshold=0.5):
    filtered_matches = []
    for m, n in matches:
        if m.distance <= dist_threshold * n.distance:
            filtered_matches.append(m)
            
    return filtered_matches

def estimate_motion(matches, kp1, kp2, k, depth1, max_depth=3000):
    
    rmat = np.eye(3)
    tvec = np.zeros((3, 1))
    
    image1_points = np.float32([kp1[m.queryIdx].pt for m in matches])
    image2_points = np.float32([kp2[m.trainIdx].pt for m in matches])
    
    cx = k[0, 2]
    cy = k[1, 2]
    fx = k[0, 0]
    fy = k[1, 1]
    
    object_points = np.zeros((0, 3))
    delete = []
    
    for i, (u, v) in enumerate(image1_points):
        z = depth1[int(round(v)), int(round(u))]
        
        if z > max_depth:
            delete.append(i)
            continue
            
        x = z * (u - cx) / fx
        y = z * (v - cy) / fy
        object_points = np.vstack([object_points, np.array([x, y, z])])
        #object_points = np.vstack([obeject_points, np.linalg.inv(k).dot(z*np.array([u, v, 1]))])
        
    image1_points = np.delete(image1_points, delete, 0)
    image2_points = np.delete(image2_points, delete, 0)
    
    _, rvec, tvec, inliers = cv2.solvePnPRansac(object_points, image2_points, k, None)
    rmat = cv2.Rodrigues(rvec)[0]
    
    return rmat, tvec, image1_points, image2_points
def visual_odometry(handler, detector='sift', matching='BF', filter_match_distance=None,
                    stereo_matcher='sgbm', mask=None, subset=None, plot=False):
    # Determine if handler has lidar data
    lidar = handler.lidar
    
    # Report methods being used to user
    print('Generating disparities with Stereo{}'.format(str.upper(stereo_matcher)))
    print('Detecting features with {} and matching with {}'.format(str.upper(detector),
                                                                  matching))
    if filter_match_distance is not None:
        print('Filtering feature matches at threshold of {}*distance'.format(filter_match_distance))
    if lidar:
        print('Improving stereo depth estimation with lidar data')
    if subset is not None:
        num_frames = subset
    else:
        num_frames = handler.num_frames
        
    if plot:
        fig = plt.figure(figsize=(14, 14))
        ax = fig.add_subplot(projection='3d')
        ax.view_init(elev=-20, azim=270)
        xs = handler.gt[:, 0, 3]
        ys = handler.gt[:, 1, 3]
        zs = handler.gt[:, 2, 3]
        ax.set_box_aspect((np.ptp(xs), np.ptp(ys), np.ptp(zs)))
        ax.plot(xs, ys, zs, c='k')
        
    # Establish a homogeneous transformation matrix. First pose is identity
    T_tot = np.eye(4)
    trajectory = np.zeros((num_frames, 3, 4))
    trajectory[0] = T_tot[:3, :]
    imheight = handler.imheight
    imwidth = handler.imwidth
    
    # Decompose left camera projection matrix to get intrinsic k matrix
    k_left, r_left, t_left = decompose_projection_matrix(handler.P0)
    
    handler.reset_frames()
    image_plus1 = next(handler.images_left)
        
    # Iterate through all frames of the sequence
    for i in range(num_frames - 1):
        start = datetime.datetime.now()
        
        image_left = image_plus1
        image_plus1 = next(handler.images_left)
        image_right = next(handler.images_right)
            
        depth = stereo_2_depth(image_left,
                               image_right,
                               P0=handler.P0,
                               P1=handler.P1,
                               matcher=stereo_matcher
                              )
        
        if lidar:

            pointcloud = next(handler.pointclouds)
            
            lidar_depth = pointcloud2image(pointcloud,
                                           imheight=imheight,
                                           imwidth=imwidth,
                                           Tr=handler.Tr,
                                           P0=handler.P0
                                          )
            indices = np.where(lidar_depth > 0)
            depth[indices] = lidar_depth[indices]
            
        # Get keypoints and descriptors for left camera image of two sequential frames
        kp0, des0 = extract_features(image_left, detector, mask)
        kp1, des1 = extract_features(image_plus1, detector, mask)
        
        # Get matches between features detected in two subsequent frames
        matches_unfilt = match_features(des0, 
                                        des1,
                                        matching=matching,
                                        detector=detector
                                       )
        #print('Number of features before filtering: ', len(matches_unfilt))
        
        # Filter matches if a distance threshold is provided by user
        if filter_match_distance is not None:
            matches = filter_matches_distance(matches_unfilt, filter_match_distance)
        else:
            matches = matches_unfilt
            
        
        #print('Number of features after filtering: ', len(matches))
        #print('Length of kp0:', len(kp0))
        #print('Length of kp1:', len(kp1))
            
        # Estimate motion between sequential images of the left camera
        rmat, tvec, img1_points, img2_points = estimate_motion(matches,
                                                               kp0,
                                                               kp1,
                                                               k_left,
                                                               depth
                                                              )
        
        # Create a blank homogeneous transformation matrix
        Tmat = np.eye(4)
        Tmat[:3, :3] = rmat
        Tmat[:3, 3] = tvec.T
        
        T_tot = T_tot.dot(np.linalg.inv(Tmat))
        
        trajectory[i+1, :, :] = T_tot[:3, :]
        print(T_tot)
        
        end = datetime.datetime.now()
        print('Time to compute frame {}:'.format(i+1), end-start)
        
        if plot:
            xs = trajectory[:i+2, 0, 3]
            ys = trajectory[:i+2, 1, 3]
            zs = trajectory[:i+2, 2, 3]
            plt.plot(xs, ys, zs, c='chartreuse')
            plt.pause(1e-32)
            
    if plot:
        plt.close()
        
    return trajectory

def main():
  handler = Dataset_Handler('00')

  # # Display the first image
  # plt.figure(figsize=(11,7))
  # plt.imshow(handler.first_image_left)
  # plt.show()

  # # Compute the left disparity map with 'bm' matcher
  # disp = compute_left_disparity_map(handler.first_image_left,
  #                                 handler.first_image_right,
  #                                 matcher='bm',
  #                                 verbose=True)
  # plt.figure(figsize=(11,7))
  # plt.imshow(disp)
  # plt.show()

  # # Compute the left disparity map with 'sgbm' matcher
  disp = compute_left_disparity_map(handler.first_image_left,
                                  handler.first_image_right,
                                  matcher='sgbm',
                                  verbose=True)
  # plt.figure(figsize=(11,7))
  # plt.imshow(disp)
  # plt.show()

  k_left, r_left, t_left = decompose_projection_matrix(handler.P0)
  k_right, r_right, t_right = decompose_projection_matrix(handler.P1)
  depth = calc_depth_map(disp, k_left, t_left, t_right)
  # plt.figure(figsize=(11,7))
  # plt.grid(False)
  # plt.imshow(depth)
  # plt.show()

  mask = np.zeros(depth.shape, dtype=np.uint8)
  ymax = depth.shape[0]
  xmax = depth.shape[1]
  cv2.rectangle(mask, (96, 0), (xmax, ymax), (255), thickness=-1)
  # plt.imshow(mask)
  # plt.show()
  
  depth = stereo_2_depth(handler.first_image_left,
                       handler.first_image_right,
                       handler.P0,
                       handler.P1,
                       matcher='sgbm',
                       rgb=False,
                       verbose=True)
  # plt.grid(False)
  # plt.imshow(depth)
  # plt.show()

  pcloud = handler.first_pointcloud
  # print('full pcloud shape', pcloud.shape)
  trimmed_pcloud = pcloud[pcloud[:, 0] > 0]
  # print('trimmed pcloud shape', trimmed_pcloud.shape)
  hom_pcloud = np.hstack([trimmed_pcloud[:, :3], np.ones(trimmed_pcloud.shape[0]).reshape((-1,1))])
  cam_xyz = handler.Tr.dot(trimmed_pcloud.T)
  cam_xyz /= cam_xyz[2]
  cam_xyz = np.vstack([cam_xyz, np.ones(cam_xyz.shape[1])])
  projection = handler.P0.dot(cam_xyz)
  projection /= projection[2]
  # print(projection[:, :5].T)

  pixel_coords = np.round(projection, 0).T.astype('int')
  # print(pixel_coords[:5])
  stereo_depth = stereo_2_depth(handler.first_image_left,
                       handler.first_image_right,
                       handler.P0,
                       handler.P1,
                       matcher='sgbm',
                       rgb=False,
                       verbose=True)
  # plt.grid(False)
  # plt.imshow(depth)
  # plt.show()
  render = pointcloud2image(handler.first_pointcloud, 
                          handler.imheight,
                          handler.imwidth,
                          handler.Tr,
                          handler.P0
                         )
  # plt.figure(figsize=(13,5))
  # plt.imshow(render)
  # plt.show()
  handler.reset_frames()
  poses = (gt for gt in handler.gt)
  pcloud_frames = (pointcloud2image(next(handler.pointclouds),
                                    handler.imheight,
                                    handler.imwidth,
                                    handler.Tr,
                                    handler.P0)
                  for i in range(handler.num_frames))

  #poses = (gt for gt in handler.gt)
  # xs = []
  # ys = []
  # zs = []
  # compute_times = []
  # fig = plt.figure()
  # ax = fig.add_subplot(projection='3d')
  # ax.view_init(elev=-20, azim=270)
  # ax.plot(handler.gt[:, 0, 3], handler.gt[:, 1, 3], handler.gt[:, 2, 3], c='k')
  # ax.set_xlabel('x')
  # ax.set_ylabel('y')
  # ax.set_zlabel('z')

  # stereo_l = handler.images_left
  # stereo_r = handler.images_right

  # for i in range(handler.num_frames // 50):
  #     img_l = next(stereo_l)
  #     img_r = next(stereo_r)
  #     start = datetime.datetime.now()
  #     disp = compute_left_disparity_map(img_l, img_r, matcher='sgbm')
  #     disp /= disp.max()
  #     #disp = 1 - disp
  #     disp = (disp*255).astype('uint8')
  #     #disp = cv2.applyColorMap(disp, cv2.COLORMAP_RAINBOW)
  #     pcloud = next(pcloud_frames)
  #     pcloud /= pcloud.max()
  #     pcloud = (pcloud*255).astype('uint8')
      
      
  #     gt = next(poses)
  #     xs.append(gt[0, 3])
  #     ys.append(gt[1, 3])
  #     zs.append(gt[2, 3])
  #     plt.plot(xs, ys, zs, c='chartreuse')
  #     plt.pause(0.000000000000000001)
  #     cv2.imshow('camera', img_l)
  #     cv2.imshow('disparity', disp)
  #     cv2.imshow('lidar', pcloud)
  #     cv2.waitKey(1)
      
  #     end = datetime.datetime.now()
  #     compute_times.append(end-start)
      
  # plt.close()
  # cv2.destroyAllWindows()
  # Using the orb features/descriptors
  image_left = handler.first_image_left
  image_right = handler.first_image_right
  image_plus1 = handler.second_image_left

  start = datetime.datetime.now()
  kp0, des0 = extract_features(image_left, 'orb', mask)
  kp1, des1 = extract_features(image_plus1, 'orb', mask)
  matches = match_features(des0, des1, matching='BF', detector='orb', sort=False)
  # print('Number of matches before filtering: ', len(matches))
  matches = filter_matches_distance(matches, 0.3)
  # print('Number of matches after filtering: ', len(matches))
  end = datetime.datetime.now()
  # print('Time to match and filter: ', end-start)
  # visualize_matches(image_left, kp0, image_plus1, kp1, matches)

  k, r, t, _, _, _, _ = cv2.decomposeProjectionMatrix(handler.P0)
  print(k)
  rmat, tvec, image1_points, image2_points = estimate_motion(matches, kp0, kp1, k, depth)
  print("Rotation Matrix:")
  print(rmat.round(4))
  print("Translation Vector:")
  print(tvec.round(4))

  transformation_matrix = np.hstack([rmat, tvec])
  print(transformation_matrix.round(4))

  trajectory_test = visual_odometry(handler, 
                                  detector='sift',
                                  matching='BF',
                                  filter_match_distance=0.3,
                                  stereo_matcher='sgbm',
                                  mask=mask,
                                  subset=None,
                                  plot=True
                                 )








main()
