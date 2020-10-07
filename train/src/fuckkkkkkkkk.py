#!/usr/bin/env python3

import threading, queue
import time
import os
import shutil
import numpy as np
import math
import rospy
import tensorflow as tf
from test_obstacle_sac import SAC
from test_obstacle_env import Test
from arm_control.arm_task import ArmTask
from manipulator_h_base_module_msgs.msg import P2PPose

MAX_EPISODES = 100000
MAX_EP_STEPS =  600
MEMORY_CAPACITY = 10000
BATCH_SIZE = 512
SIDE = ['right_', 'left_']
SIDE_ = ['R', 'L']
GOAL_REWARD = 800
LOAD = False
SAVE = [False, False]
COUNTER = [1, 1]
TRAIN_CNT = [0, 0]
EP = [0, 0]
WORKS = 1
SUCCESS_ARRAY = np.zeros([2,500])
GOAL_RATE = [40, 40]
ACTION_FLAG = [False, False]

def worker(name, workers, agent):
    SUCCESS_ARRAY = np.zeros([1000])
    SUCCESS_ARRAY_P2P = np.zeros([1000])
    SUCCESS_ARRAY_LINE = np.zeros([1000])
    COLLISION_ARRAY = np.zeros([1000])
    COLLISION_ARRAY_P2P = np.zeros([1000])
    COLLISION_ARRAY_LINE = np.zeros([1000])
    IKFAIL_ARRAY = np.zeros([1000])
    IKFAIL_ARRAY_P2P = np.zeros([1000])
    IKFAIL_ARRAY_LINE = np.zeros([1000])
    SINGULARITY_ARRAY = np.zeros([1000])
    SINGULARITY_ARRAY_P2P = np.zeros([1000])
    SINGULARITY_ARRAY_LINE = np.zeros([1000])
    S_RATE = 0
    S_RATE_P2P = 0
    S_RATE_LINE = 0
    C_RATE = 0
    C_RATE_P2P = 0
    C_RATE_LINE = 0
    I_RATE = 0
    I_RATE_P2P = 0
    I_RATE_LINE = 0
    COLLISION = False
    IKFAIL = False
    SINGULARITY = False
    arm = ArmTask(SIDE[name]+'arm')
    env = Test(name, workers) #0 = right
    time.sleep(0.5)

    print(threading.current_thread())
    print('name', name, 'workers', workers, 'agentID', id(agent))
    reset_start = False

    if RUN_FLAG[workers].is_set():
        RUN_FLAG[workers].clear()
    else:
        RUN_FLAG[workers].set()
    RUN_FLAG[workers].wait()
   
    t1 = time.time()
    while (not COORD.should_stop()) and (not rospy.is_shutdown()):
        if RUN_FLAG[workers].is_set():
            RUN_FLAG[workers].clear()
        else:
            RUN_FLAG[workers].set()
            
        RUN_FLAG[workers].wait()
        # RUN_FLAG[workers].set()
    # print("fuckkkkkkkkkkkkkkkkkkkkkkkkk")
    # while (not COORD.should_stop()) and (not rospy.is_shutdown()):
    #     print("aaaaaaaaaaaaaaaa")
    #     if RUN_FLAG[workers].is_set():
    #         RUN_FLAG[workers].clear()
    #         print("ddddddddddddddddddda")
    #     else:
    #         RUN_FLAG[workers].set()
    #         print("dyyyyyyyyyyyyyyyyyyyyddda")
    #     RUN_FLAG[workers].wait()


        # rate = rospy.Rate(30)
        for cnt in range(1000):


            done_cnt = 0
            COLLISION = False
            IKFAIL = False
            SINGULARITY = False

            # if name == 0:
            #     WORKER_EVENT[1].set()
            # else:
            #     WORKER_EVENT[0].set()
                
            time.sleep(1)
            s = env.reset(reset_start)
            if name == 1:
                time.sleep(0.5)
            reset_start = True
            goal = env.get_goal
            goal = np.append(goal, 0)
            start = (s[:8])
            for __ in range(1000):
                # WORKER_EVENT[name].wait()

                a = agent.choose_action(s)
                s, done, collision, ik_success, singularity = env.step(a)
                done_cnt += int(done)
                if collision:
                    COLLISION_ARRAY[cnt%1000] = 1
                    break
                # if COLLISION and collision:
                #     COLLISION_ARRAY[cnt%1000] = 1
                #     break
                # elif collision:
                #     COLLISION = True
                # else:
                #     COLLISION = False
                if IKFAIL and not ik_success:
                    IKFAIL_ARRAY[cnt%1000] = 1
                elif not ik_success:
                    IKFAIL = True
                else:
                    IKFAIL = False
                if __ > 5:
                    if SINGULARITY:
                        SINGULARITY_ARRAY[cnt%1000] = 1
                    elif singularity:
                        SINGULARITY = True
                if done_cnt > 0:    
                    SUCCESS_ARRAY[cnt%1000] = 1
                    reset_start = True
                    # COLLISION_ARRAY[cnt%1000] = 0
                    # IKFAIL_ARRAY[cnt%1000] = 0
                    break
                if __ == 999:
                    reset_start = True
                # rate.sleep()


            # WORKER_EVENT[nameIndx].clear()
            




            # arm.clear_cmd()
            # COLLISION = False
            # IKFAIL = False
            # SINGULARITY = False
            # time.sleep(0.5)
            # env.move_arm(start)
            # time.sleep(1)
            # arm.ikMove_quat('p2p', goal[:3], goal[3:7], goal[7])
            # while arm.is_busy:
            #     if env.check_collision():
            #         COLLISION = True
            #         COLLISION_ARRAY_P2P[cnt%1000] = 1
            #     if arm.is_ikfail:
            #         IKFAIL = True
            #         IKFAIL_ARRAY_P2P[cnt%1000] = 1
            #         arm.clear_cmd()
            #         break
            #     if arm.singularity:
            #         SINGULARITY = True
            #         SINGULARITY_ARRAY_P2P[cnt%1000] = 1
            # if not COLLISION and not IKFAIL:
            #     SUCCESS_ARRAY_P2P[cnt%1000] = 1
            # COLLISION = False
            # IKFAIL = False
            # SINGULARITY = False
            # time.sleep(0.5)
            # env.move_arm(start)
            # time.sleep(1)
            # arm.ikMove_quat('line', goal[:3], goal[3:7], goal[7])
            # while arm.is_busy:
            #     if env.check_collision():
            #         COLLISION = True
            #         COLLISION_ARRAY_LINE[cnt%1000] = 1
            #     if arm.is_ikfail:
            #         IKFAIL = True
            #         IKFAIL_ARRAY_LINE[cnt%1000] = 1
            #         arm.clear_cmd()
            #         break
            #     if arm.singularity:
            #         SINGULARITY = True
            #         SINGULARITY_ARRAY_LINE[cnt%1000] = 1
            # if not COLLISION and not IKFAIL:
            #     SUCCESS_ARRAY_LINE[cnt%1000] = 1

            S_RATE = 0
            S_RATE_P2P = 0
            S_RATE_LINE = 0
            C_RATE = 0
            C_RATE_P2P = 0
            C_RATE_LINE = 0
            I_RATE = 0
            I_RATE_P2P = 0
            I_RATE_LINE = 0
            SI_RATE = 0
            SI_RATE_P2P = 0
            SI_RATE_LINE = 0
            for z in SUCCESS_ARRAY:
                S_RATE += z
            for z in SUCCESS_ARRAY_P2P:
                S_RATE_P2P += z
            for z in SUCCESS_ARRAY_LINE:
                S_RATE_LINE += z
            for z in COLLISION_ARRAY:
                C_RATE += z
            for z in COLLISION_ARRAY_P2P:
                C_RATE_P2P += z
            for z in COLLISION_ARRAY_LINE:
                C_RATE_LINE += z
            for z in IKFAIL_ARRAY:
                I_RATE += z
            for z in IKFAIL_ARRAY_P2P:
                I_RATE_P2P += z
            for z in IKFAIL_ARRAY_LINE:
                I_RATE_LINE += z
            for z in SINGULARITY_ARRAY:
                SI_RATE += z
            for z in SINGULARITY_ARRAY_P2P:
                SI_RATE_P2P += z
            for z in SINGULARITY_ARRAY_LINE:
                SI_RATE_LINE += z
            print('Ep:', cnt, 's_rate:', S_RATE, S_RATE_P2P, S_RATE_LINE, '    c_rate:', C_RATE, C_RATE_P2P, C_RATE_LINE, \
                        '    i_rate', I_RATE, I_RATE_P2P, I_RATE_LINE, '    si_rate:', SI_RATE, SI_RATE_P2P, SI_RATE_LINE)


def train(name):
    global SAVE, COUNTER, RUN_FLAG, cmd, move
    threads_ = []
    print(threading.current_thread())
        
    
    env = Test(name, 0) #0 = right

    agent = SAC(act_dim=env.act_dim, obs_dim=env.obs_dim, name=SIDE[name])
    arm = ArmTask(SIDE[name]+'arm')

    print('name', name, 'agentID', id(agent))
    env = None
    for j in range(WORKS):
        t = threading.Thread(target=worker, args=(name, j, agent,))
        threads_.append(t)
        if name == 0:
            RUN_FLAG.append(threading.Event())
            RUN_FLAG[j].set()
    time.sleep(1)
    for i in threads_:
        i.start()


def right_callback(msg):
    global cmd, move
    cmd[0][0] = msg.pose.position.x
    cmd[0][1] = msg.pose.position.y
    cmd[0][2] = msg.pose.position.z
    cmd[0][3] = msg.pose.orientation.w
    cmd[0][4] = msg.pose.orientation.x
    cmd[0][5] = msg.pose.orientation.y
    cmd[0][6] = msg.pose.orientation.z
    move[0] = True

def left_callback(msg):
    global cmd, move
    cmd[1][0] = msg.pose.position.x
    cmd[1][1] = msg.pose.position.y
    cmd[1][2] = msg.pose.position.z
    cmd[1][3] = msg.pose.orientation.w
    cmd[1][4] = msg.pose.orientation.x
    cmd[1][5] = msg.pose.orientation.y
    cmd[1][6] = msg.pose.orientation.z
    move[1] = True





if __name__ == '__main__':
    rospy.init_node('aL')
    threads = []
    cmd = np.zeros([2,7])
    move = [False, False]
    RUN_FLAG = []

    # LEARN_EVENT = [threading.Event(), threading.Event()]
    # WORKER_EVENT = [threading.Event(), threading.Event()]
    COORD = tf.train.Coordinator()
    
    for i in range(2):
        t = threading.Thread(target=train, args=(i,))
        threads.append(t)
        # WORKER_EVENT[i].set()
        # LEARN_EVENT[i].clear()

    for i in threads:
        i.start()
        time.sleep(10)
    print(threading.current_thread())
    COORD.join(threads)

    rospy.Subscriber('right_arm/drl_pose_msg', P2PPose, right_callback)
    rospy.Subscriber('left_arm/drl_pose_msg', P2PPose, left_callback)
    rospy.spin()
