#!/usr/bin/env python3

import numpy as np
import cv2
import roslib
import rospy
import struct
import math
import time
from std_msgs.msg import Float32MultiArray, MultiArrayLayout, Float32MultiArray, MultiArrayDimension, String
from std_msgs.msg import Header
from sensor_msgs import point_cloud2
from sensor_msgs.msg import Image, PointCloud2, PointField
from std_srvs.srv import *
# from rcnn_pkg.srv import get_mask, get_maskResponse
# import ros_numpy
from rcnn_pkg.srv import graymask
from sensor_msgs.msg import CameraInfo, CompressedImage
from geometry_msgs.msg import PoseArray, PoseStamped
from visualization_msgs.msg import Marker, MarkerArray
import rospkg
from cv_bridge import CvBridge, CvBridgeError
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import os 
import message_filters

import detectron2
from detectron2.utils.logger import setup_logger

# import some common detectron2 utilities
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.engine import DefaultTrainer
from detectron2 import model_zoo
from detectron2.data.datasets import register_coco_instances

flag_do_prediction = True



class rcnn_detection(object):
	def __init__(self):
		
		r = rospkg.RosPack()
		self.path = r.get_path('rcnn_pkg')
		self.cv_bridge = CvBridge() 
		
		#### Publisher
		self.image_pub = rospy.Publisher("~predict_img", Image, queue_size = 1)
		self.predict_mask_pub = rospy.Publisher("prediction_mask", Image, queue_size = 1)
		self.predict_mask_jet_pub = rospy.Publisher("/prediction_mask_jet", Image, queue_size = 1)
		self.predict_pc_pub = rospy.Publisher("/prediction_pc", PointCloud2, queue_size=1)
		self.obj_id = rospy.Publisher("~predict_obj_id", String, queue_size = 1)

		#### Service
		self.service = rospy.Service('id_prediction_mask', graymask, self.id_prediction_mask)
		# rospy.Service("id_prediction_mask", prediction_mask, self.c)
		# rospy.Service('get_maskcloud', get_mask, self.get_maskcloud)

		################ �o�̥����n��tableware dataset ####################
		# register_coco_instances('delta_val', {}, 
		# 						'/home/arg/detectron2/datasets/class_5_val/annotations.json', 
		# 					'/home/arg/detectron2/datasets/class_5_val')
		# self.subt_metadata = MetadataCatalog.get("delta_val")
		# self.dataset_dicts = DatasetCatalog.get("delta_val")
		################ �i��令���U�o�� �A�̦ۦ�d�N�@�U���| ####################
		register_coco_instances('tableware', {}, 
								'/home/arg/Mask-RCNN-Detectron2/datasets/tableware/annotations.json', 
							'/home/arg/Mask-RCNN-Detectron2/datasets/tableware')
		self.subt_metadata = MetadataCatalog.get("tableware")
		self.dataset_dicts = DatasetCatalog.get("tableware")



		self.cfg = get_cfg()
		self.cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
		self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo

		################ �o�̥����n��NUM_CLASSES �]�tbackground �ҥH�|�O3 class + 1 ####################
		self.cfg.MODEL.ROI_HEADS.NUM_CLASSES = 14  # datasets classes

		self.cfg.DATALOADER.NUM_WORKERS = 0 #Single thread



		################ �o�̥����n�� WEIGHTS ####################
		# self.cfg.MODEL.WEIGHTS = os.path.join(self.path, "weights", "model_0096989.pth")
		self.cfg.MODEL.WEIGHTS = os.path.join(self.path, "weights", "model_0028999.pth")

		self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set the testing threshold for this model
		self.predictor = DefaultPredictor(self.cfg)

		### msg filter 
		# image_sub = message_filters.Subscriber('/ex_side_camera/color/image_raw', Image)
		# depth_sub = message_filters.Subscriber('/ex_side_camera/aligned_depth_to_color/image_raw', Image)
		self.target_id = -1
		image_sub = message_filters.Subscriber('/camera/color/image_raw', Image)
		depth_sub = message_filters.Subscriber('/camera/aligned_depth_to_color/image_raw', Image)
		ts = message_filters.TimeSynchronizer([image_sub, depth_sub], 10)
		ts.registerCallback(self.callback)

		rospy.Service('maskrcnn_prediction', SetBool, self.prediction_handler)
		# rospy.Service('get_maskcloud', get_mask, self.get_maskcloud)


	def prediction_handler(self, req):
		global flag_do_prediction
		print ("%s"%req.data)

		if req.data == False:
			flag_do_prediction = False
			return [True, "Stop prediction"]

		if req.data == True:
			flag_do_prediction = True
			return [True, "Start prediction"]
	
	def id_prediction_mask(self, req):
		print(req.obj_id)
		

		try:
			if "wine_glass" == req.obj_id:
				self.target_id=13
				print("wine_glass")			
			elif "water_glass" == req.obj_id:
				self.target_id=12
				print("water_glass")
			elif "teacup" == req.obj_id:
				self.target_id=11
				print("teacup")			
			elif "small_spoon" == req.obj_id:
				self.target_id=10
				print("small_spoon")
			elif "small_plates" == req.obj_id:
				self.target_id=9
				print("small_plates")			
			elif "small_fork" == req.obj_id:
				self.target_id=8
				print("small_fork")
			elif "packet_cup'" == req.obj_id:
				self.target_id=7
				print("packet_cup'")			
			elif "mug" == req.obj_id:
				self.target_id=6
				print("mug")			
			elif "knife" == req.obj_id:
				self.target_id=5
				print("knife")
			elif "bowl" == req.obj_id:
				self.target_id=4
				print("bowl")			
			elif "big_spoon" == req.obj_id:
				self.target_id=3
				print("big_spoon")
			elif "big_plates" == req.obj_id:
				self.target_id=2
				print("big_plates")
			elif "big_fork" == req.obj_id:
				self.target_id=1
				print("big_fork")
			# if "spoon" == req.obj_id:
			# 	self.target_id=3
			# 	print("spoon")
			# elif "knife" == req.obj_id:
			# 	self.target_id=2
			# 	print("knife")
			# elif "fork" == req.obj_id:
			# 	self.target_id=1
			# 	print("fork")
		except:
			print("fail")
			# res.result = "Fail"
			# return "Fail"

		# area_one = np.asarray(pred_one)*10
		# gray_img = np.uint8(area)
		# gray_img_one = np.uint8(area_one)

		# backtorgb = cv2.applyColorMap(gray_img, cv2.COLORMAP_JET)
		# cv_rgbmask = self.cv_bridge.cv2_to_imgmsg(backtorgb)
		# self.predict_mask_jet_pub.publish(cv_rgbmask)
		# # gray3_img = cv2.cvtColor(gray_img,cv2.COLOR_GRAY2RGB)
		# # cv_graymask = self.cv_bridge.cv2_to_imgmsg(gray_img)
		# cv_graymask = self.cv_bridge.cv2_to_imgmsg(gray_img_one)

		# res.obj_mask = cv_graymask
		# cv2.imwrite("mask.png", gray_img_one)

		return True

	def callback(self, img_msg, depth):
		if flag_do_prediction == True :
			try:
				cv_image = self.cv_bridge.imgmsg_to_cv2(img_msg, "bgr8")
			except CvBridgeError as e:
				print(e)

			outputs = self.predictor(cv_image)	
			v = Visualizer(cv_image[:, :, ::-1],
						metadata=self.subt_metadata, 
						scale=0.8, 
					#	 instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels
			)
			v = v.draw_instance_predictions(outputs["instances"].to("cpu"))

			self.image_pub.publish(self.cv_bridge.cv2_to_imgmsg(v.get_image()[:, :, ::-1], "bgr8"))
			print("Detected 1 frame !!!")

			pred_classes = outputs["instances"].pred_classes.cpu().numpy()
			pred_masks = outputs["instances"].pred_masks.cpu().numpy().astype(float)
			pred = pred_masks.copy()
			pred_all = np.zeros([480, 640], np.float)
			pred_one = np.zeros([480, 640], np.float)

			#################################### �̭��I������ ���A�n��predict mask �M��output point cloud############################################
			# # pred_classes: 1: fork, 2: knife, 3: spoon, 4: waterproof, 5: gear
			for i in range(len(pred_classes)):
				pred[i] = pred_classes[i]*pred_masks[i]
				pred_all += pred[i]

			area = np.asarray(pred_all)*10

			print(type(pred_classes))
			if (13 == self.target_id) and (13 in pred_classes):
				print("wine_glass")
				j, = np.where(np.isclose(pred_classes, 3))
				self.obj_id.publish("12")
				pred_one = pred_classes[j[0]]*pred_masks[j[0]]
			elif (12 == self.target_id) and (12 in pred_classes):
				print("water_glass")
				j, = np.where(np.isclose(pred_classes, 12))
				self.obj_id.publish("11")
				pred_one = pred_classes[j[0]]*pred_masks[j[0]]
			elif (11 == self.target_id) and (11 in pred_classes):
				print("teacup")
				j, = np.where(np.isclose(pred_classes, 11))
				self.obj_id.publish("10")
				pred_one = pred_classes[j[0]]*pred_masks[j[0]]
			elif (10 == self.target_id) and (10 in pred_classes):
				print("small_spoon")
				j, = np.where(np.isclose(pred_classes, 10))
				self.obj_id.publish("9")
				pred_one = pred_classes[j[0]]*pred_masks[j[0]]
			elif (9 == self.target_id) and (9 in pred_classes):
				print("small_plates")
				j, = np.where(np.isclose(pred_classes, 9))
				self.obj_id.publish("8")
				pred_one = pred_classes[j[0]]*pred_masks[j[0]]
			elif (8 == self.target_id) and (8 in pred_classes):
				print("small_fork")
				j, = np.where(np.isclose(pred_classes, 8))
				self.obj_id.publish("7")
				pred_one = pred_classes[j[0]]*pred_masks[j[0]]
			elif (7 == self.target_id) and (7 in pred_classes):
				print("packet_cup'")
				j, = np.where(np.isclose(pred_classes, 7))
				self.obj_id.publish("6")
				pred_one = pred_classes[j[0]]*pred_masks[j[0]]
			elif (6 == self.target_id) and (6 in pred_classes):
				print("mug")
				j, = np.where(np.isclose(pred_classes, 6))
				self.obj_id.publish("5")
				pred_one = pred_classes[j[0]]*pred_masks[j[0]]
			elif (5 == self.target_id) and (5 in pred_classes):
				print("knife")
				j, = np.where(np.isclose(pred_classes, 5))
				self.obj_id.publish("4")
				pred_one = pred_classes[j[0]]*pred_masks[j[0]]
			elif (4 == self.target_id) and (4 in pred_classes):
				print("bowl")
				j, = np.where(np.isclose(pred_classes, 4))
				self.obj_id.publish("3")
				pred_one = pred_classes[j[0]]*pred_masks[j[0]]
			elif (3 == self.target_id) and (3 in pred_classes):
				print("big_spoon")
				j, = np.where(np.isclose(pred_classes, 3))
				self.obj_id.publish("2")
				pred_one = pred_classes[j[0]]*pred_masks[j[0]]
			elif 2 == self.target_id and (2 in pred_classes):
				print("big_plates")
				j, = np.where(np.isclose(pred_classes, 2))
				self.obj_id.publish("1")
				pred_one = pred_classes[j[0]]*pred_masks[j[0]]
			elif 1 == self.target_id and (1 in pred_classes):
				print("big_fork")
				j, = np.where(np.isclose(pred_classes, 1))
				self.obj_id.publish("0")
				pred_one = pred_classes[j[0]]*pred_masks[j[0]]
			else:
				print("nothing")
				return			

			# # if 5 in pred_classes:
			# # 	j, = np.where(np.isclose(pred_classes, 5))
			# # 	self.obj_id.publish("4")
			# # 	pred_one = pred_classes[j[0]]*pred_masks[j[0]]
			# # elif 4 in pred_classes:
			# # 	j, = np.where(np.isclose(pred_classes, 4))
			# # 	self.obj_id.publish("3")
			# # 	pred_one = pred_classes[j[0]]*pred_masks[j[0]]
			# if (3 == self.target_id) and (3 in pred_classes):
			# 	print("spoon")
			# 	j, = np.where(np.isclose(pred_classes, 3))
			# 	self.obj_id.publish("2")
			# 	pred_one = pred_classes[j[0]]*pred_masks[j[0]]
			# elif 2 == self.target_id and (2 in pred_classes):
			# 	print("knife")
			# 	j, = np.where(np.isclose(pred_classes, 2))
			# 	self.obj_id.publish("1")
			# 	pred_one = pred_classes[j[0]]*pred_masks[j[0]]
			# elif 1 == self.target_id and (1 in pred_classes):
			# 	print("fork")
			# 	j, = np.where(np.isclose(pred_classes, 1))
			# 	self.obj_id.publish("0")
			# 	pred_one = pred_classes[j[0]]*pred_masks[j[0]]
			# else:
			# 	print("nothing")
			# 	return False
			
			area_one = np.asarray(pred_one)*10
			gray_img = np.uint8(area)
			gray_img_one = np.uint8(area_one)

			backtorgb = cv2.applyColorMap(gray_img, cv2.COLORMAP_JET)
			cv_rgbmask = self.cv_bridge.cv2_to_imgmsg(backtorgb)
			self.predict_mask_jet_pub.publish(cv_rgbmask)
			# gray3_img = cv2.cvtColor(gray_img,cv2.COLOR_GRAY2RGB)
			# cv_graymask = self.cv_bridge.cv2_to_imgmsg(gray_img)
			cv_graymask = self.cv_bridge.cv2_to_imgmsg(gray_img_one)

			self.predict_mask_pub.publish(cv_graymask)
			# print(type(gray_img_one))
			cv2.imwrite("mask.png", gray_img_one)
			# self.get_maskcloud(cv_image, gray_img, depth)

	
	# def get_maskcloud(self, rgb_img, gray_img, depth_msg):
	# 	depth = self.cv_bridge.imgmsg_to_cv2(depth_msg)
	# 	# res = get_maskResponse()
	# 	points = []
	# 	# rgb_image = self.cv_bridge.imgmsg_to_cv2(global_rgb_data, "bgr8")
	# 	# depth = self.cv_bridge.imgmsg_to_cv2(global_depth_data)
	# 	# mask = prediction_save(rgb_image)

	# 	cx = 312.89288330078125 #323.6322
	# 	cy = 245.52340698242188 #240.377166
	# 	fx = 474.853271484375 #607.2167
	# 	fy = 474.8533020019531 #607.34753

	# 	for v in range(rgb_img.shape[1]):
	# 		for u in range(rgb_img.shape[0]):
	# 			color = rgb_img[u,v].astype('float32')
				
	# 			Z = depth[u,v] / 1000.0
	# 			if Z==0: continue
	# 			X = (v - cx) * Z / fx
	# 			Y = (u - cy) * Z / fy
	# 			if(gray_img[u,v] != 0):
	# 				points.append([X,Y,Z,color[0],color[1],color[2]])
	# 	points = np.array(points)

	# 	fields = [PointField('x', 0, PointField.FLOAT32, 1), \
	# 			  PointField('y', 4, PointField.FLOAT32, 1), \
	# 			  PointField('z', 8, PointField.FLOAT32, 1), \
	# 			  PointField('r', 12, PointField.FLOAT32, 1),\
	# 			  PointField('g', 16, PointField.FLOAT32, 1),\
	# 			  PointField('b', 20, PointField.FLOAT32, 1),\
	# 			  ]
	# 	header = Header()
	# 	header.frame_id = "camera_link"
	# 	pc2 = point_cloud2.create_cloud(header, fields, points)
	# 	pc2.header.stamp = rospy.Time.now()
	# 	self.predict_pc_pub.publish(pc2) 

	def onShutdown(self):
		rospy.loginfo("Shutdown.")	
	

if __name__ == '__main__': 
	rospy.init_node('rcnn_detection',anonymous=False)
	rcnn_detection = rcnn_detection()
	rospy.on_shutdown(rcnn_detection.onShutdown)
	rospy.spin()






# thing_classes=['_background_', 'big_fork', 'big_plates', 'big_spoon', 'bowl', 'knife', 'mug', 'packet_cup', 'small_fork', 'small_plates', 'small_spoon', 'teacup', 'water_glass', 'wine_glass'],
# thing_dataset_id_to_contiguous_id={0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 10: 10, 11: 11, 12: 12, 13: 13})
