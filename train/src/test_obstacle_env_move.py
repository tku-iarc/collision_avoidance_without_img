#!/usr/bin/env python3

"""classic Acrobot task"""
import sys
sys.path.insert(1, "/home/yue/.local/lib/python3.5/site-packages/")
sys.path.insert(2, "/home/yue/yuetin/collision_surrouding/catkin_workspace/install/lib/python3/dist-packages")
# sys.path.insert(1, "/home/yue/.local/lib/python3.5/site-packages/")
# sys.path.insert(1, "/usr/local/lib/python3.5/dist-packages/")
# sys.path.insert(0, '/opt/installer/open_cv/cv_bridge/lib/python3/dist-packages/')




# sys.path.insert('/opt/ros/kinetic/lib/python2.7/dist-packages')
import numpy as np
from DNN_v2 import *
from numpy import sin, cos, pi
from gym import core, spaces
from gym.utils import seeding
import rospy
import math
import random
import time
from train.srv import get_state, move_cmd, set_goal, set_start
from train.msg import aa_box_pos
# from train import aa_box_pos_msg
from CheckCollision_v1 import CheckCollision
from gazebo_msgs.msg import ModelState
from std_msgs.msg import String,Float32MultiArray
# from CheckCollision_tensor import CheckCollision
# from vacuum_cmd_msg.srv import VacuumCmd
import cv2




from cv_bridge import CvBridge, CvBridgeError
from cv_bridge.boost.cv_bridge_boost import getCvType
from sensor_msgs.msg import Image,CompressedImage
from nav_msgs.msg import Odometry
from gazebo_msgs.msg import ContactsState,ContactState
# sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
# import sys

# sys.path.insert(2, "/home/yue/yuetin/collision_surrouding/catkin_workspace/install/lib/python3/dist-packages")

# sys.path.insert(1, "/usr/local/lib/python3.5/dist-packages/")



class Test(core.Env):
    # ACTION_VEC_TRANS = 1/600
    # ACTION_ORI_TRANS = 1/100
    # ACTION_PHI_TRANS = 1/100
    ACTION_VEC_TRANS = 1/180
    ACTION_ORI_TRANS = 1/60
    ACTION_PHI_TRANS = 1/60

    NAME = ['/right_', '/left_', '/right_']

    def __init__(self, name, workers):
        self.__name = self.NAME[name%2]
        self.__obname = self.NAME[name%2 + 1]
        # if workers == 0:
        self.workers = 'arm'
        # else:
        #     self.workers = str(workers)

        high = np.array([1.,1.,1.,1.,1.,1.,1.,1.,  #8
                         1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,     #7
                         0.,0.,0.,0.,0.,0.,         #6
                         0.,0.,0.,0.,0.,0.,0.,0.,0.,#9
                         0.,0.,0.,0.,0.,0.,0.,0.,0.,                  #2
                         0.,0.,0.,0.,0.,0.])                 #1
                                                    #24
        low = -1*high 
                    # ox,oy,oz,oa,ob,oc,od,of,
                    # vx,vy,vz,va,vb,vc,vd,vf
                    # dis of Link(15)
                    # 
                    # joint_angle(7), 
                    # limit(1), rate(3)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        self.action_space = spaces.Discrete(3)
        self.act_dim=8
        self.obs_dim=57+24+2
        self.state = []
        self.action = []
        self.cmd = []
        self.point = []
        self.goal = []
        self.goal_pos = []
        self.goal_quat = []
        self.goal_phi = 0
        self.goal_rpy = []
        self.old = []
        self.old_pos = []
        self.old_quat = []
        self.old_phi = 0
        self.joint_pos = []
        self.joint_angle = []
        self.limit = []
        self.s_jointpos = []
        # self.dis_pos
        self.cc = CheckCollision()
        self.collision = False
        self.range_cnt = 0.6#+(workers*0.1)
        self.rpy_range = 0.2*(workers+1)
        self.done = True
        self.s_cnt = 0
        self.goal_err = 0.08
        self.ori_err = 0.3
        self.quat_inv = False
        self.goal_angle = []
        self.object_pub = 0
        self.set_model_pub = rospy.Publisher(
            '/gazebo/set_model_state',
            ModelState,
            queue_size=1,
            latch=True
        )
        self.set_mode_pub = rospy.Publisher(
            self.__name+self.workers+'/set_mode_msg',
            String,
            queue_size=1,
            latch=True
        )

        self.set_aabox_pub = rospy.Publisher(
            'aa_box_pos',
            aa_box_pos,
            queue_size=1,
            latch=True
        )



        self.bridge = CvBridge()
        self.depth_dim = 76800
        self.depth_image = None
        self.image_input = None
        self.__bumper = None
        self.__robot = "_arm"
        self.images_ = []
        
        
        self.aa_box_x = 0
        self.aa_box_y = 0
        # self.set_mode_pub.publish('set_mode')
        

        rospy.Subscriber('/ir_depth/depth/image_raw',Image,self.callback )
        self.seed(9)
        # self.seed(345*(workers+1) + 467*(name+1))
        self.reset(True)

        ## image (gazebo)

        
        # rospy.Subscriber("odom", Odometry,self.get_aa_box_position)
        # rospy.Subscriber("/bumper",ContactsState,self.Sub_Bumper)
    
        self.arm_move = rospy.Publisher(
            self.__name + 'arm_move',
            Float32MultiArray,
            queue_size=1,
            latch=True
            )
        
    @property
    def get_goal(self):
        return self.goal


    @property
    def is_success(self):
        return self.done

    @property
    def success_cnt(self):
        return self.s_cnt
        
    ## image (gazebo)

    # def get_aa_box_position(self,msg = None):
    #     print(msg.pose.pose.position.x)
    #     print(msg.pose.pose.position.y)
    def aa_suck(self,data):
        try:    
            self.aa_box_x = data.x
            self.aa_box_y = data.y


        except CvBridgeError as e:
            print(e)

    # def set_aa_box_vel(self, goal_x, goal_y):
        
    def callback(self,data):
        try:
            # if len(data.data) == 0:
            #     print("data.data == 0")
            tmp = self.bridge.imgmsg_to_cv2(data,"32FC1")
            tmp = cv2.resize(tmp , (320,240))
            where_are_nan = np.isnan(tmp)
            where_are_inf = np.isinf(tmp)
            tmp[where_are_nan] = 10
            tmp[where_are_inf] = 10
            # print(tmp)
            self.images_ = tmp / 10
            # print("aaas")
            # if self.images_.shape == (0,):
            #     rospy.logwarn("self.images_ = 0")
            # print("aaaaaaaaaaa")
            # print(self.images_)
            # cv2.imshow('My Image', self.images_)
            # cv2.waitKey(1)


        except CvBridgeError as e:
            print(e)



    def get_state_client(self, name):
        service = name+self.workers+'/get_state'
        try:
            rospy.wait_for_service(service, timeout=1.)
        except rospy.ROSException as e:
            rospy.logwarn('wait_for_service timeout')
            self.get_state_client(name)
            
        client = rospy.ServiceProxy(
            service,
            get_state
        )
        # res = client(cmd)
        res = client.call()
        return res

    def move_cmd_client(self, cmd, name):
        service = name+self.workers+'/move_cmd'
        try:
            rospy.wait_for_service(service, timeout=1.)
        except rospy.ROSException as e:
            rospy.logwarn('wait_for_service timeout')
            self.move_cmd_client(cmd, name)
            
        client = rospy.ServiceProxy(
            service,
            move_cmd
        )
        # res = client(cmd)
        res = client.call(cmd)
        return res

    def set_start_client(self, cmd, rpy, name):
        service = name+self.workers+'/set_start'
        try:
            rospy.wait_for_service(service, timeout=1.)
        except rospy.ROSException as e:
            rospy.logwarn('wait_for_service timeout')
            self.set_start_client(cmd, rpy, name)
            
        client = rospy.ServiceProxy(
            service,
            set_start
        )
        # res = client(cmd)
        res = client(action=cmd, rpy=rpy)
        return res

    def set_goal_client(self, cmd, rpy, name):
        service = name+self.workers+'/set_goal'
        try:
            rospy.wait_for_service(service, timeout=1.)
        except rospy.ROSException as e:
            rospy.logwarn('wait_for_service timeout')
            self.set_goal_client(cmd, rpy, name)
            
        client = rospy.ServiceProxy(
            service,
            set_goal
        )
        # res = client(cmd)
        res = client(action=cmd, rpy=rpy)
        return res

    def set_object(self, name, pos, ori):
        msg = ModelState()
        msg.model_name = name
        msg.pose.position.x = pos[0]
        msg.pose.position.y = pos[1]
        msg.pose.position.z = pos[2]
        msg.pose.orientation.w = ori[0]
        msg.pose.orientation.x = ori[1]
        msg.pose.orientation.y = ori[2]
        msg.pose.orientation.z = ori[3]
        msg.reference_frame = 'world'
        self.set_model_pub.publish(msg)

    def set_object2(self, name, pos, ori):
        msg = ModelState()
        msg.model_name = name+self.workers
        msg.pose.position.x = pos[0]
        msg.pose.position.y = pos[1]
        msg.pose.position.z = pos[2]
        msg.pose.orientation.w = ori[0]
        msg.pose.orientation.x = ori[1]
        msg.pose.orientation.y = ori[2]
        msg.pose.orientation.z = ori[3]
        msg.reference_frame = 'world'
        self.set_model_pub.publish(msg)


    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    def quat_angle(self):
        cos_q = np.dot(self.goal[3:7],  self.old[3:7])
        if (self.quat_inv and cos_q>0) or ((not self.quat_inv) and cos_q<=0):
            cos_q = np.dot(-1*self.goal[3:7],  self.old[3:7])
        return math.acos(cos_q)/pi


    def move(self, goal):
        self.goal = goal
        res = self.get_state_client(self.__name)
        res_ = self.get_state_client(self.__obname)
        self.old, self.joint_pos[:15], self.joint_angle = np.array(res.state), res.joint_pos, res.joint_angle
        self.limit, self.goal_quat, self.quat_inv, self.joint_pos[15:30] = res.limit, res.quaterniond, res.quat_inv, res_.joint_pos
        linkPosM, linkPosS = self.collision_init()
        alarm, Link_dis = self.cc.checkCollision(linkPosM, linkPosS)
        s = np.append(self.old[:3], np.subtract(self.goal[:3], self.old[:3]))
        s = np.append(s, Link_dis)
        s = np.append(s, self.joint_angle)
        s = np.append(s, self.limit[0])
        self.dis_pos = np.linalg.norm(self.goal[:3] - s[:3])
        s = np.append(s, self.dis_pos)
        self.state = s
        return s

    def reset(self, reset_start):
        # self.old, self.joint_pos[:15], self.joint_pos[15:30], self.joint_angle, self.limit, self.goal_quat, self.quat_inv = self.set_old()
        # linkPosM, linkPosS = self.collision_init()
        # alarm, Link_dis = self.cc.checkCollision(linkPosM, linkPosS)
        # alarm_cnt = 0
        # for i in alarm:
        #     alarm_cnt += i
        # if alarm_cnt>0:
        #     return self.reset()
        # joint_pos_tmp = self.joint_pos
        # self.goal, self.goal_angle, self.joint_pos[:15], self.joint_pos[15:30]= self.set_goal()
        # linkPosM, linkPosS = self.collision_init()
        # alarm, Link_dis = self.cc.checkCollision(linkPosM, linkPosS)
        # alarm_cnt = 0
        # for i in alarm:
        #     alarm_cnt += i
        # if alarm_cnt>0:
        #     return self.reset()


        if reset_start:
            self.old, self.joint_pos[:15], self.joint_pos[15:30], self.joint_angle, self.limit, self.goal_quat, self.quat_inv = self.set_old()
            linkPosM, linkPosS = self.collision_init()
            alarm, Link_dis = self.cc.checkCollision(linkPosM, linkPosS)
            alarm_cnt = 0
            for i in alarm:
                alarm_cnt += i
            if alarm_cnt>0:
                return self.reset(reset_start)

        joint_pos_tmp = self.joint_pos
        self.goal, self.goal_angle, self.joint_pos[:15], self.joint_pos[15:30]= self.set_goal()
        linkPosM, linkPosS = self.collision_init()
        alarm, Link_dis = self.cc.checkCollision(linkPosM, linkPosS)
        alarm_cnt = 0
        for i in alarm:
            alarm_cnt += i
        self.joint_pos = joint_pos_tmp


        self.state = np.append(self.old, np.subtract(self.goal[:3], self.old[:3]))
        self.state = np.append(self.state, self.goal_quat[:4])
        # self.state = np.append(self.state, np.subtract(self.goal[3:7], self.old[3:7]))
        # self.state = np.append(self.state, np.subtract(-1*self.goal[3:7], self.old[3:7]))
        self.state = np.append(self.state, Link_dis)
        self.state = np.append(self.state, self.joint_angle)
        self.state = np.append(self.state, self.limit)
        self.dis_pos = np.linalg.norm(self.goal[:3] - self.old[:3])
        self.dis_ori = math.sqrt(np.linalg.norm(self.goal[3:7] - self.old[3:7]) + np.linalg.norm(-1*self.goal[3:7] - self.old[3:7]) - 2)
        # self.angle_ori = self.quat_angle()
        self.state = np.append(self.state, self.dis_pos)
        self.state = np.append(self.state, self.dis_ori)
        self.state = np.append(self.state, self.joint_pos[6:12])
        self.state = np.append(self.state, self.goal_angle)



        # print("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
        # print(self.images_)
        # self.img_suckkkkkkkkkkkkk = np.reshape(self.images_,-1)
        # print("aaa")
        # print(self.img_suckkkkkkkkkkkkk.shape)
        # print("bbb")
        # self.state = np.append(self.state, self.img_suckkkkkkkkkkkkk)
        
        if self.__name == '/right_':
            # self.aa_box_x = random.uniform(0.15,0.3)
            # self.aa_box_y = random.uniform(-0.5,0.5)
            self.aa_box_x = random.uniform(0.1,0.3)
            self.aa_box_y = random.uniform(-0.5,0.5)
            # self.aa_box_y = random.uniform(-0.5,-0.1)
            self.set_object('table_box', (0.55,0,0.345), (0, 0, 0, 0))
            self.set_object('aa_box', (self.aa_box_x,self.aa_box_y,0.8), (0, 0, 0, 0))
            aa_box = aa_box_pos()
            aa_box.x = self.aa_box_x
            aa_box.y = self.aa_box_y
            self.set_aabox_pub.publish(aa_box)
            # rospy.Publisher('aa_box_pos',aa_box_pos)
            # print(self.aa_box_x,"III",self.aa_box_x,self.aa_box_y)
        # self.set_object2(self.__name, (self.goal[0]-0.08, self.goal[1], self.goal[2]+1.45086), (0, 0, 0, 0))
        time.sleep(0.5)
        linkPosM, linkPosS = self.collision_init()
        alarm, Link_dis = self.cc.checkCollision(linkPosM, linkPosS)
        alarm_cnt = 0
        for i in alarm:
            alarm_cnt += i
        if alarm_cnt>0:
            return self.reset(reset_start)
        arm_x = 0.08
        arm_r_y = 1
        arm_l_y = -1
        arm_z = -1.4125
        if self.goal[0]<self.aa_box_x+0.05+arm_x and self.aa_box_x-0.05+arm_x<self.goal[0] and self.goal[1]<(self.aa_box_y+0.05)*arm_r_y and (self.aa_box_y+0.05)*arm_r_y<self.goal[1]:
            return self.reset(reset_start)
        rospy.Subscriber('aa_box_pos',aa_box_pos,self.aa_suck)
        # if self.__name == '/right_':
        #     self.aa_box_possition = [ self.aa_box_x+0.05+arm_x, (self.aa_box_y+0.05)*arm_r_y, 0.975+arm_z + 0.125,
        #                             self.aa_box_x+0.05+arm_x, (self.aa_box_y-0.05)*arm_r_y, 0.975+arm_z + 0.125,
        #                             self.aa_box_x-0.05+arm_x, (self.aa_box_y+0.05)*arm_r_y, 0.975+arm_z + 0.125,
        #                             self.aa_box_x-0.05+arm_x, (self.aa_box_y-0.05)*arm_r_y, 0.975+arm_z + 0.125,
        #                             self.aa_box_x+0.05+arm_x, (self.aa_box_y+0.05)*arm_r_y, 0.625+arm_z,
        #                             self.aa_box_x+0.05+arm_x, (self.aa_box_y-0.05)*arm_r_y, 0.625+arm_z,
        #                             self.aa_box_x-0.05+arm_x, (self.aa_box_y+0.05)*arm_r_y, 0.625+arm_z, 
        #                             self.aa_box_x-0.05+arm_x, (self.aa_box_y-0.05)*arm_r_y, 0.625+arm_z]     

        # if self.__name == '/left_':
        #     self.aa_box_possition = [ self.aa_box_x+0.05+arm_x, (self.aa_box_y+0.05)*arm_l_y, 0.975+arm_z + 0.125,
        #                             self.aa_box_x+0.05+arm_x, (self.aa_box_y-0.05)*arm_l_y, 0.975+arm_z + 0.125,
        #                             self.aa_box_x-0.05+arm_x, (self.aa_box_y+0.05)*arm_l_y, 0.975+arm_z + 0.125,
        #                             self.aa_box_x-0.05+arm_x, (self.aa_box_y-0.05)*arm_l_y, 0.975+arm_z + 0.125,
        #                             self.aa_box_x+0.05+arm_x, (self.aa_box_y+0.05)*arm_l_y, 0.625+arm_z,
        #                             self.aa_box_x+0.05+arm_x, (self.aa_box_y-0.05)*arm_l_y, 0.625+arm_z,
        #                             self.aa_box_x-0.05+arm_x, (self.aa_box_y+0.05)*arm_l_y, 0.625+arm_z, 
        #                             self.aa_box_x-0.05+arm_x, (self.aa_box_y-0.05)*arm_l_y, 0.625+arm_z]

        if self.__name == '/right_':
            self.aa_box_possition = [ self.aa_box_pos_x+0.05, (self.aa_box_pos_y+0.05), self.aa_box_pos_z + 0.3 ,
                                    self.aa_box_pos_x+0.05, (self.aa_box_pos_y-0.05), self.aa_box_pos_z + 0.3 ,
                                    self.aa_box_pos_x-0.05, (self.aa_box_pos_y+0.05), self.aa_box_pos_z + 0.3 ,
                                    self.aa_box_pos_x-0.05, (self.aa_box_pos_y-0.05), self.aa_box_pos_z + 0.3 ,
                                    self.aa_box_pos_x+0.05, (self.aa_box_pos_y+0.05), self.aa_box_pos_z-0.3,
                                    self.aa_box_pos_x+0.05, (self.aa_box_pos_y-0.05), self.aa_box_pos_z-0.3,
                                    self.aa_box_pos_x-0.05, (self.aa_box_pos_y+0.05), self.aa_box_pos_z-0.3, 
                                    self.aa_box_pos_x-0.05, (self.aa_box_pos_y-0.05), self.aa_box_pos_z-0.3]     

        if self.__name == '/left_':
            self.aa_box_possition = [ self.aa_box_pos_x+0.05, (self.aa_box_pos_y+0.05), self.aa_box_pos_z + 0.3 ,
                                    self.aa_box_pos_x+0.05, (self.aa_box_pos_y-0.05), self.aa_box_pos_z + 0.3 ,
                                    self.aa_box_pos_x-0.05, (self.aa_box_pos_y+0.05), self.aa_box_pos_z + 0.3 ,
                                    self.aa_box_pos_x-0.05, (self.aa_box_pos_y-0.05), self.aa_box_pos_z + 0.3 ,
                                    self.aa_box_pos_x+0.05, (self.aa_box_pos_y+0.05), self.aa_box_pos_z-0.3,
                                    self.aa_box_pos_x+0.05, (self.aa_box_pos_y-0.05), self.aa_box_pos_z-0.3,
                                    self.aa_box_pos_x-0.05, (self.aa_box_pos_y+0.05), self.aa_box_pos_z-0.3, 
                                    self.aa_box_pos_x-0.05, (self.aa_box_pos_y-0.05), self.aa_box_pos_z-0.3] 

        self.state = np.append(self.state, self.aa_box_possition)


        aa_box_possssss = np.array([self.aa_box_x,self.aa_box_y,0.8])
        self.dis_obstacle_1 = np.linalg.norm(aa_box_possssss[:3] - self.joint_pos[6:9])
        self.dis_obstacle_2 = np.linalg.norm(aa_box_possssss[:3] - self.joint_pos[9:12])


        self.state = np.append(self.state, self.dis_obstacle_1)
        self.state = np.append(self.state, self.dis_obstacle_2)


        self.collision = False
        self.done = False
        self.success = False
        return self.state

    def set_goal(self):
        self.goal = self.np_random.uniform(low=-0.5, high=0.5, size=(8,))
        self.goal[1] = self.np_random.uniform(low=0.1, high=0.5)
        # self.goal = self.np_random.uniform(low=-0.5, high=0.5, size=(8,))
        # self.goal[1] = self.np_random.uniform(low=0.1, high=0.35)        
        
        # self.goal[1] = 0.3
        if self.__name == '/left_':
            self.goal[1] = self.np_random.uniform(low=-0.5, high=-0.1)
        self.goal[2] = self.np_random.uniform(low=-0.3, high=0.5)
        rpy = self.np_random.uniform(low=-1*self.rpy_range, high=self.rpy_range, size=(4,))
        # if self.__name == '/left_':
        #     self.goal[1] = self.np_random.uniform(low=-0.35, high=-0.1)
        # self.goal[2] = self.np_random.uniform(low=-0.2, high=0.5)
        # rpy = self.np_random.uniform(low=-1*self.rpy_range, high=self.rpy_range, size=(4,))
        
        
        
        # print('self.goal = ', self.goal)
        # if self.goal[0]>0.5: 
        #     if self.goal[0]>0.75:
        #         self.goal[2] /= -3
        #     else:
        #         self.goal[2] /= -2
        #     self.goal[2]+=1

        self.goal[0] = 0
        # self.goal[3] = self.range_cnt/2
        self.goal = np.append(self.goal, self.range_cnt)
        res = self.set_goal_client(self.goal, rpy, self.__name)
        res_ = self.get_state_client(self.__obname)
        goal_pos = np.array(res.state)
        if not res.success:
            return self.set_goal()
        else:
            return goal_pos[:7], res.joint_angle, res.joint_pos, res_.joint_pos

    def set_old(self):
        # self.start = self.np_random.uniform(low=-0.5, high=0.5, size=(8,))
        # rpy = self.np_random.uniform(low=-1*self.rpy_range, high=self.rpy_range, size=(4,))
        
        self.start = self.np_random.uniform(low=-0.5, high=0.5, size=(8,))
        self.start[1] = self.np_random.uniform(low=0.1, high=0.5)
        if self.__name == '/left_':
            self.start[1] = self.np_random.uniform(low=-0.5, high=-0.1)
        self.start[2] = self.np_random.uniform(low=-0.3, high=0.5)
        rpy = self.np_random.uniform(low=-1*self.rpy_range, high=self.rpy_range, size=(4,))        
        
        
        
        # if self.start[0]>0.5:
        #     if self.start[0]>0.75:
        #         self.start[2] /= -3
        #     else:
        #         self.start[2] /= -2
        #     self.start[2] +=1
        self.start[0] = 0
        # self.start[3] = self.range_cnt/2
        self.start = np.append(self.start, self.range_cnt)
        res = self.set_start_client(self.start, rpy, self.__name)
        res_ = self.get_state_client(self.__obname)
        old_pos = np.array(res.state)
        if not res.success:
            return self.set_old()
        else:
            return old_pos, res.joint_pos, res_.joint_pos, res.joint_angle, res.limit, res.quaterniond, res.quat_inv

    def collision_init(self):
        linkPosM = np.array(self.joint_pos[:15])
        linkPosS = np.array(self.joint_pos[15:])
        linkPosM = np.append([0.,0.,-0.8], linkPosM)
        linkPosS = np.append([0.,0.,-0.8], linkPosS)
        linkPosM = linkPosM.reshape(6,3)
        linkPosS = linkPosS.reshape(6,3)
        return linkPosM, linkPosS

    def move_arm(self, cmd):
        self.move_cmd_client(cmd, self.__name)

    def check_collision(self):
        res = self.get_state_client(self.__name)
        res_ = self.get_state_client(self.__obname)
        self.joint_pos[:15], self.joint_pos[15:30] = res.joint_pos, res_.joint_pos
        linkPosM, linkPosS = self.collision_init()
        alarm, _ = self.cc.checkCollision(linkPosM, linkPosS)
        alarm_cnt = 0
        for i in alarm:
            alarm_cnt += i
        if alarm_cnt>0.4:
            return True
        else:
            return False


    def step(self, a):
        alarm = []
        Link_dis = []
        bumpalarm = []
        s = self.state
        suck = self.image_input

        self.collision = False
        action_vec = a[:3]*self.ACTION_VEC_TRANS
        action_ori = a[3:7]*self.ACTION_ORI_TRANS
        action_phi = a[7]*self.ACTION_PHI_TRANS
        self.action = np.append(action_vec, action_ori)
        self.action = np.append(self.action, action_phi)
        self.cmd = np.add(s[:8], self.action)
        self.cmd[3:7] /= np.linalg.norm(self.cmd[3:7])

        res = self.move_cmd_client(self.cmd, self.__name)
        res_ = self.get_state_client(self.__obname)
        if res.success:
            self.old, self.joint_pos[:15], self.joint_angle = np.array(res.state), res.joint_pos, res.joint_angle
            self.limit, self.goal_quat, self.quat_inv, self.joint_pos[15:30] = res.limit, res.quaterniond, res.quat_inv, res_.joint_pos
            linkPosM, linkPosS = self.collision_init()
            alarm, Link_dis = self.cc.checkCollision(linkPosM, linkPosS)
            s = np.append(self.old, np.subtract(self.goal[:3], self.old[:3]))
            s = np.append(s, self.goal_quat[:4])
            # s = np.append(s, np.subtract(self.goal[3:7], self.old[3:7]))
            # s = np.append(s, np.subtract(-1*self.goal[3:7], self.old[3:7]))
            s = np.append(s, Link_dis)
            s = np.append(s, self.joint_angle)
            s = np.append(s, self.limit)
            self.dis_pos = np.linalg.norm(self.goal[:3] - s[:3])
            self.dis_ori = math.sqrt(np.linalg.norm(self.goal[3:7] - s[3:7]) + np.linalg.norm(-1*self.goal[3:7] - s[3:7]) - 2)
            # self.angle_ori = self.quat_angle()
            s = np.append(s, self.dis_pos)
            s = np.append(s, self.dis_ori)
            s = np.append(s, self.joint_pos[6:12])
            s = np.append(s, self.goal_angle)


            rospy.Subscriber('aa_box_pos',aa_box_pos,self.aa_suck)
            # print(self.aa_box_x,"III",self.aa_box_y)
            # if self.__name == '/right_':
            self.set_object('table_box', (0.55,0,0.345), (0, 0, 0, 0))
            self.set_object('aa_box', (self.aa_box_x,self.aa_box_y,0.82691), (0, 0, 0, 0))
            # self.image_input = np.reshape(self.images_,-1)
            # print(self.image_cnn[10])
            




            arm_x = 0.08
            arm_r_y = 1
            arm_l_y = -1
            arm_z = -1.4125
            
            if self.__name == '/right_':
                self.aa_box_possition = [ self.aa_box_x+0.05+arm_x, (self.aa_box_y+0.05)*arm_r_y, 0.975+arm_z+0.125,
                                        self.aa_box_x+0.05+arm_x, (self.aa_box_y-0.05)*arm_r_y, 0.975+arm_z+0.125,
                                        self.aa_box_x-0.05+arm_x, (self.aa_box_y+0.05)*arm_r_y, 0.975+arm_z+0.125,
                                        self.aa_box_x-0.05+arm_x, (self.aa_box_y-0.05)*arm_r_y, 0.975+arm_z+0.125,
                                        self.aa_box_x+0.05+arm_x, (self.aa_box_y+0.05)*arm_r_y, 0.625+arm_z,
                                        self.aa_box_x+0.05+arm_x, (self.aa_box_y-0.05)*arm_r_y, 0.625+arm_z,
                                        self.aa_box_x-0.05+arm_x, (self.aa_box_y+0.05)*arm_r_y, 0.625+arm_z, 
                                        self.aa_box_x-0.05+arm_x, (self.aa_box_y-0.05)*arm_r_y, 0.625+arm_z]     

            if self.__name == '/left_':
                self.aa_box_possition = [ self.aa_box_x+0.05+arm_x, (self.aa_box_y+0.05)*arm_l_y, 0.975+arm_z,
                                        self.aa_box_x+0.05+arm_x, (self.aa_box_y-0.05)*arm_l_y, 0.975+arm_z,
                                        self.aa_box_x-0.05+arm_x, (self.aa_box_y+0.05)*arm_l_y, 0.975+arm_z,
                                        self.aa_box_x-0.05+arm_x, (self.aa_box_y-0.05)*arm_l_y, 0.975+arm_z,
                                        self.aa_box_x+0.05+arm_x, (self.aa_box_y+0.05)*arm_l_y, 0.625+arm_z,
                                        self.aa_box_x+0.05+arm_x, (self.aa_box_y-0.05)*arm_l_y, 0.625+arm_z,
                                        self.aa_box_x-0.05+arm_x, (self.aa_box_y+0.05)*arm_l_y, 0.625+arm_z, 
                                        self.aa_box_x-0.05+arm_x, (self.aa_box_y-0.05)*arm_l_y, 0.625+arm_z]  
                                        

            s = np.append(s, self.aa_box_possition)
            aa_box_possssss = np.array([self.aa_box_x,self.aa_box_y,0.8])
            self.dis_obstacle_1 = np.linalg.norm(aa_box_possssss[:3] - self.joint_pos[6:9])
            self.dis_obstacle_2 = np.linalg.norm(aa_box_possssss[:3] - self.joint_pos[9:12])
            # suck = np.append(suck, self.image_input)

            s = np.append(s, self.dis_obstacle_1)
            s = np.append(s, self.dis_obstacle_2)
            # s = np.append(s, self.image_input)
            # print(suck)
            # print(s)

        
        terminal = self._terminal(s, res.success, alarm)
        alarm_cnt = 0
        # for i in bumpalarm:
        #     alarm_cnt += i
        # if alarm_cnt>0.4:
        #     terminal = True
        # reward = self.get_reward(s, res.success, terminal, res.singularity)
        # self.images_ = suck
        if (not self.collision) and math.fabs(s[7])<0.9:
            self.state = s


        ## see
        if self.workers == 'arm':
            if self.object_pub == 0:
                self.set_object2(self.__name, (self.goal[0]-0.08, self.goal[1], self.goal[2]+1.45086), (0, 0, 0, 0))
                self.object_pub = 1
            else:
                self.set_object2(self.__name+'q', (self.goal[0]-0.08, self.goal[1], self.goal[2]+1.45086), self.goal[3:7])
                self.object_pub = 0

        fail = False
        # if not res.success or self.collision or res.singularity:
        #     fail = True
        linkPosM, linkPosS = self.collision_init()
        alarm, Link_dis = self.cc.checkCollision(linkPosM, linkPosS)
        alarm_cnt = 0
        for i in alarm:
            alarm_cnt += i
        if alarm_cnt>0.4:
            fail = True
            self.collision = True


        # return self.state, reward, terminal, self.success, fail
        return self.state, terminal, self.collision, res.success, res.singularity



    def _terminal(self, s, ik_success, alarm):
        alarm_cnt = 0
        for i in alarm:
            alarm_cnt += i
        if alarm_cnt>0.4:
            self.collision = True
        if ik_success and not self.collision:
            if self.dis_pos < 0.05:
                if not self.done:
                    self.done = True
                    self.s_cnt += 1
                return True
            else:
                return False
        else:
            return False