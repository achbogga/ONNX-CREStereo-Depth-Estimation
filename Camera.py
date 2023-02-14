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

	def calculate_depth_maps(self, disparity_map) -> None:
		"""
		Description:
			Calculate the Depth Maps from stereo aligned with the left camera and also right cameras
		Args:
			None
		Returns:
			None
		"""
		# left aligned
		fx = self.left_cam.fx
		self.depth_map_left_aligned = np.zeros((self.image_width, self.image_height), dtype=float)
		self.depth_map_left_aligned = fx*self.baseline/disparity_map

		# right aligned
		fx = self.right_cam.fx
		self.depth_map_right_aligned = np.zeros((self.image_width, self.image_height), dtype=float)
		self.depth_map_right_aligned = fx*self.baseline/disparity_map
		
	def get_point_cloud_left_aligned(self, disparity_map) -> np.ndarray:
		"""
		Description:
			Calculate the Relative Point clouds aligned with left camera
		Args:
			disparity_map -> np.ndarray
		Returns:
			left_points -> np.ndarray
		"""
		K_left = self.left_cam.get_intrinsic_matrix()
		M_stereo_extrinsic = self.get_stereo_transform()
		Q_left = np.matmul(K_left, M_stereo_extrinsic)
		left_points = cv2.reprojectImageTo3D(disparity_map, Q_left)
		return left_points
	
	def get_point_cloud_right_aligned(self, disparity_map) -> np.ndarray:
		"""
		Description:
			Calculate the Relative Point clouds aligned with the right camera
		Args:
			disparity_map -> np.ndarray
		Returns:
			right_points -> np.ndarray
		"""
		K_right = self.right_cam.get_intrinsic_matrix()
		M_stereo_extrinsic = self.get_stereo_transform()
		Q_right = np.matmul(K_right, M_stereo_extrinsic)
		right_points = cv2.reprojectImageTo3D(disparity_map, Q_right)
		return right_points



