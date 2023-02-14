import pyzed.sl as sl
import typing as t
import numpy as np
import cv2

class DistortionParameters:
	def __init__(self, disto: np.ndarray) -> None:
		self.disto = disto
		"""
		A Numpy array.
		Distortion factor : [ k1, k2, p1, p2, k3 ]. Radial (k1,k2,k3) and Tangential (p1,p2) distortion.
		"""
		

class PinholeCamera:
	def __init__(self, 
				fx: float, 
				fy: float, 
				cx: float, 
				cy: float, 
				image_size: t.Tuple[t.Any],
				disto: DistortionParameters = None, 
				v_fov: float = None, 
				h_fov: float = None,
				d_fov: float = None,
				) -> None:
		"""
		A PinholeCamera class to hold all the camera intrinsic parameters
		Args:
			fx : Focal length in pixels along x axis. 
			fy : Focal length in pixels along y axis. 
			cx : Optical center along x axis, defined in pixels (usually close to width/2). 
			cy : Optical center along y axis, defined in pixels (usually close to height/2). 
			disto : A Distortion factor object.
			v_fov : Vertical feild of view, in degrees.
			h_fov: Horizontal field of view, in degrees.
			d_fov: Diagonal field of view, in degrees.
			image_size: Image resolution (Width, Height) in pixels
		Returns:
			None
		"""
		vars(self).update((k,v) for k,v in vars().items()
                   if k != 'self' and k not in vars(self))
		self.calculate_intrinsic_matrix()
		
	def calculate_intrinsic_matrix(self,) -> None:
		"""
		Calculate the internal parameters as intrinsic matrix K in the homogenous co-ordinates
		Args:
			None
		Returns:
			None
		"""
		K = np.zeros((3, 4), dtype=float)
		K[0, 0] = self.fx
		K[1, 1] = self.fy
		K[2, 2] = 1.0
		K[0, 2] = self.cx
		K[1, 2] = self.cy
		self.K = K
	
	def get_intrinsic_matrix(self, ) -> np.ndarray:
		"""
		Get the intrinsic Matrix K (3,4)
		Args:
			None
		Returns:
			np.ndarray
		"""
		return self.K


class StereoCamera:
	def __init__(self, 
				left_cam: PinholeCamera, 
				right_cam: PinholeCamera, 
				stereo_baseline: float, 
				max_dist_in_meters = 2.0,
				) -> None:
		"""
		Description:
			A generic class template for Stereo Camera
		Args:
			left_cam: The left camera intinsic parameters defined as PinholeCamera object
						
			right_cam: The left camera intinsic parameters defined as PinholeCamera object
			stereo_baseline: The distance between both the eyes of the stereo camera
			stereo_transform: The camera rotation and translation matrix from right to left using left as reference
		Returns:
			None
		"""
		self.left_cam = left_cam
		self.right_cam = right_cam
		try:
			assert left_cam.image_size == right_cam.image_size
		except AssertionError as AE:
			print ('left_dims: ', left_cam.image_size, 'right_dims: ', right_cam.image_size)
			raise AE
		self.image_width, self.image_height = left_cam.image_size
		self.stereo_baseline = stereo_baseline
	
	def set_translation(self, 
						translation_vector: np.ndarray (shape=(3,), dtype=float), 
					   ):
		"""
		Description:
			sets the translation vector with the given array
		Args:
			translation_vector: Translation between the two stereo eyes. [tx, ty, tz]
		Returns:
			None
		"""
		self.translation_vector = translation_vector

	def set_rotation(self, 
						rotation_matrix: np.ndarray (shape=(3,3), dtype=float), 
					   ):
		"""
		Description:
			sets the rotation matrix with the given array
		Args:
			rotation_matrix: Rotation matrix between the two stereo eyes. R
		Returns:
			None
		"""
		self.rotation_matrix = rotation_matrix

	def calculate_stereo_transform(self) -> None:
		"""
		Description:
			Calculate the Relative Position and Orientation between the two cameras in homogenous co-ordinates
		Args:
			None
		Returns:
			None
		"""
		M = np.zeros((4, 4), dtype=float)
		M[0:3, 0:3] = self.rotation_matrix
		M[:3,3] = self.translation_vector
		M[3,3] = 1.0
		self.M = M

	def get_stereo_transform(self) -> np.ndarray:
		"""
		Get the relative stereo transform Matrix M from right to left (4,4) in homogenous co-ordinates
		Args:
			None
		Returns:
			np.ndarray
		"""
		return self.M

	def get_depth_map_left_aligned(self) -> np.ndarray:
		"""
		Get the depth map left aligned
		Args:
			None
		Returns:
			np.ndarray
		"""
		return self.depth_map_left_aligned
	
	def get_depth_map_right_aligned(self) -> np.ndarray:
		"""
		Get the depth map right aligned
		Args:
			None
		Returns:
			np.ndarray
		"""
		return self.depth_map_right_aligned

	def calculate_depth_maps(self) -> None:
		"""
		Description:
			Calculate the Depth Maps from stereo aligned with the left camera and also right cameras
		Args:
			None
		Returns:
			None
		"""
		# left aligned
		self.depth_map_left_aligned = np.zeros((self.image_width, self.image_height), dtype=float)
		fx = self.left_cam.fx
		fy = self.left_cam.fy
		disp = self.disparity_map_left_aligned
		self.depth_map_left_aligned = fx*self.baseline/disp

		# right aligned
		self.depth_map_right_aligned = np.zeros((self.image_width, self.image_height), dtype=float)
		fx = self.right_cam.fx
		fy = self.right_cam.fy
		disp = self.disparity_map_right_aligned
		self.depth_map_right_aligned = fx*self.baseline/disp
		
	def calculate_point_clouds(self) -> None:
		"""
		Description:
			Calculate the Relative Point clouds aligned with both the cameras
		Args:
			None
		Returns:
			None
		"""
		left_points = cv2.reprojectImageTo3D(self.disparity_map_left_aligned, 
				       						self.get_stereo_transform())
		right_points = cv2.reprojectImageTo3D(self.disparity_map_left_aligned, 
				       						self.get_stereo_transform())

	def get_point_cloud_left_aligned(self) -> np.ndarray:
		"""
		Get the relative point cloud left camera aligned
		Args:
			None
		Returns:
			np.ndarray
		"""
		return self.left_points
	
	def get_point_cloud_right_aligned(self) -> np.ndarray:
		"""
		Get the relative point cloud left camera aligned
		Args:
			None
		Returns:
			np.ndarray
		"""
		return self.right_points


def test_with_zed():
	# Create a Camera object
	zed = sl.Camera()

	# Create a InitParameters object and set configuration parameters
	init_params = sl.InitParameters()
	init_params.depth_mode = sl.DEPTH_MODE.NEURAL  # Use one of PERFORMANCE, NEURAL, ULTRA, QUALITY depth modes
	init_params.coordinate_units = sl.UNIT.METER  # Use meter units (for depth measurements)
	init_params.camera_resolution = sl.RESOLUTION.HD720

	# Open the camera
	err = zed.open(init_params)
	if err != sl.ERROR_CODE.SUCCESS:
		exit(1)

	calibration_params = zed.get_camera_information().camera_configuration.calibration_parameters
	# Focal length of the left eye in pixels
	focal_left_x = calibration_params.left_cam.fx
	stereo_baseline = calibration_params.get_camera_baseline()
	stereo_transform = calibration_params.stereo_transform
	stereo_T = calibration_params.T
	stereo_R = calibration_params.R
	print ('focal_left_x in px: ', focal_left_x)
	print ('stereo_baseline: ', stereo_baseline)
	print ('stereo_transform: ', type(stereo_transform), stereo_transform)
	print ('stereo_T: ', type(stereo_T), stereo_T)
	print ('stereo_R: ', type(stereo_R), stereo_R)

	# Close the camera
	zed.close()

test_with_zed()
