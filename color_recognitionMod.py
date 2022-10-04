#!/usr/bin/env python
# -*- coding: utf-8 -*-



#__________________________________________________________________

from curses import ERR
from tkinter import MOVETO
import cv2
import sys
#from matplotlib import Color
import rospy
import math
import random
import threading
import numpy as np
import moveit_commander
from cv_bridge import CvBridge
from sensor_msgs.msg import CompressedImage

from xarm_msgs.srv import ColorPosition
from xarm_msgs.srv import ColorPositionRequest
from xarm_msgs.srv import ColorPositionResponse

#__________________________________________________________________

if sys.version_info < (3, 0):
    PY3 = False
    import Queue as queue
else:
    PY3 = True
    import queue
    
#__________________________________________________________________HSV COLOR STRUCTURE
COLOR_DICT = {
    'red': {'lower': np.array([0, 43, 46]), 'upper': np.array([10, 255, 255])},
    'blue': {'lower': np.array([90, 50, 70]), 'upper': np.array([128, 255, 255])},
    'green': {'lower': np.array([36, 50, 70]), 'upper': np.array([89, 255, 255])},
    'yellow': {'lower': np.array([25, 50, 70]), 'upper': np.array([34, 255, 255])},
}
#__________________________________________________________________

class GripperCtrl(object):
    def __init__(self):
        self._commander = moveit_commander.move_group.MoveGroupCommander('xarm_gripper')
        self._init()

    def _init(self):
        self._commander.set_max_acceleration_scaling_factor(1.0) #0-1 range
        self._commander.set_max_velocity_scaling_factor(1.0) #0-1 range
    
    def open(self, wait=True):
        try:
            self._commander.set_named_target('open')
            ret = self._commander.go(wait=wait)
            print('gripper_open, ret={}'.format(ret))
            return ret
        except Exception as e:
            print('[Ex] gripper open exception, {}'.format(e))
        return False

    def close(self, wait=True):
        try:
            self._commander.set_named_target('close')
            ret = self._commander.go(wait=wait)
            print('gripper_close, ret={}'.format(ret))
            return ret
        except Exception as e:
            print('[Ex] gripper close exception, {}'.format(e))
        return False

#__________________________________________________________________
class XArmCtrl(object):
    def __init__(self, dof):
        self._commander = moveit_commander.move_group.MoveGroupCommander('xarm{}'.format(dof))
        self.dof = int(dof)
        self._init()
    
    def _init(self):
        self._commander.set_max_acceleration_scaling_factor(1.0)
        self._commander.set_max_velocity_scaling_factor(1.0)

    def set_joints(self, angles, wait=True):
        try:
            joint_target = self._commander.get_current_joint_values()
            for i in range(joint_target):
                if i >= len(angles):
                    break
                if angles[i] is not None:
                    joint_target[i] = math.radians(angles[i])
            print('set_joints, joints={}'.format(joint_target))
            self._commander.set_joint_value_target(joint_target)
            ret = self._commander.go(wait=wait)
            print('move to finish, ret={}'.format(ret))
            return ret
        except Exception as e:
            print('[Ex] set_joints exception, ex={}'.format(e))
    
    def set_joint(self, angle, inx=-1, wait=True):
        try:
            joint_target = self._commander.get_current_joint_values()
            joint_target[inx] = math.radians(angle)
            print('set_joints, joints={}'.format(joint_target))
            self._commander.set_joint_value_target(joint_target)
            ret = self._commander.go(wait=wait)
            print('move to finish, ret={}'.format(ret))
            return ret
        except Exception as e:
            print('[Ex] set_joint exception, ex={}'.format(e))
        return False

    def moveto(self, x=None, y=None, z=None, ox=None, oy=None, oz=None, relative=False, wait=True):
        if x == 0 and y == 0 and z == 0 and ox == 0 and oy == 0 and oz == 0 and relative:
            return True
        try:
            pose_target = self._commander.get_current_pose().pose
            if relative:
                pose_target.position.x += x / 1000.0 if x is not None else 0
                pose_target.position.y += y / 1000.0 if y is not None else 0
                pose_target.position.z += z / 1000.0 if z is not None else 0
                pose_target.orientation.x += ox if ox is not None else 0
                pose_target.orientation.y += oy if oy is not None else 0
                pose_target.orientation.z += oz if oz is not None else 0
            else:
                pose_target.position.x = x / 1000.0 if x is not None else pose_target.position.x
                pose_target.position.y = y / 1000.0 if y is not None else pose_target.position.y
                pose_target.position.z = z / 1000.0 if z is not None else pose_target.position.z
                pose_target.orientation.x = ox if ox is not None else pose_target.orientation.x
                pose_target.orientation.y = oy if oy is not None else pose_target.orientation.y
                pose_target.orientation.z = oz if oz is not None else pose_target.orientation.z
            print('move to position=[{:.2f}, {:.2f}, {:.2f}], orientation=[{:.6f}, {:.6f}, {:.6f}]'.format(
                pose_target.position.x * 1000.0, pose_target.position.y * 1000.0, pose_target.position.z * 1000.0,
                pose_target.orientation.x, pose_target.orientation.y, pose_target.orientation.z
            ))
            if self.dof == 7:
                path, fraction = self._commander.compute_cartesian_path([pose_target], 0.005, 0.0)
                if fraction < 0.9:
                    ret = False
                else:
                    ret = self._commander.execute(path, wait=wait)
                print('move to finish, ret={}, fraction={}'.format(ret, fraction))
            else:
                self._commander.set_pose_target(pose_target)
                ret = self._commander.go(wait=wait)
                print('move to finish, ret={}'.format(ret))
            return ret
        except Exception as e:
            print('[Ex] moveto exception: {}'.format(e))
        return False

#__________________________________________________________________

class GazeboMotionThread(threading.Thread):

    def __init__(self, que, **kwargs):
        if PY3:
            super().__init__()
        else:
            super(GazeboMotionThread, self).__init__()
        self.que = que
        self.daemon = True
        self.in_motion = True
        dof = kwargs.get('dof', 6)
        self._xarm_ctrl = XArmCtrl(dof)
        self._gripper_ctrl = GripperCtrl()
        self._grab_z = kwargs.get('grab_z', 10)
        self._safe_z = kwargs.get('safe_z', 100)
    
    @staticmethod
    def _rect_to_move_params(rect):
        return int((466 - rect[0][1]) * 900.0 / 460.0 + 253.3), int((552 - rect[0][0]) * 900.0 / 460.0 - 450), rect[2] - 90
       
    def run(self):
        def cube1R(self):
            #cube1R
            self._xarm_ctrl.moveto(x=395, y=119, z=200)
            
            self._xarm_ctrl.moveto(x=395, y=119, z=16)
            
            self._gripper_ctrl.close()
            self._xarm_ctrl.moveto(x=395, y=119, z=200)
        def cube2R(self):
            #cube2R
            self._xarm_ctrl.moveto(x=440, y=-184, z=200)
            
            self._xarm_ctrl.moveto(x=440, y=-184, z=15)
            
            self._gripper_ctrl.close()
            self._xarm_ctrl.moveto(x=440, y=-184, z=200)
        def cube1B(self):
            #cube1B	
            self._xarm_ctrl.moveto(x=350, y=-190, z=200)
            
            self._xarm_ctrl.moveto(x=350, y=-190, z=15)
            
            self._gripper_ctrl.close()
            self._xarm_ctrl.moveto(x=350, y=-190, z=200)
        def cube2B(self):
            #cube2B	
            self._xarm_ctrl.moveto(x=500, y=10, z=200)
            
            self._xarm_ctrl.moveto(x=500, y=10, z=15)
            
            self._gripper_ctrl.close()
            self._xarm_ctrl.moveto(x=500, y=10, z=200) 


        def pile1(self):
            #pile1
                    
            self._xarm_ctrl.moveto(x=200, y=200, z=200)
            self._xarm_ctrl.moveto(x=500, y=200, z=200)
            self._xarm_ctrl.moveto(x=600, y=200, z=100)
            
            self._xarm_ctrl.moveto(x=650, y=200, z=16)
            
            self._gripper_ctrl.open()     
            self._xarm_ctrl.moveto(x=650, y=200, z=50)
        def pile2(self):
            #pile2
                    
            self._xarm_ctrl.moveto(x=200, y=200, z=200)
            self._xarm_ctrl.moveto(x=500, y=200, z=200)
            self._xarm_ctrl.moveto(x=600, y=200, z=150)
            
            self._xarm_ctrl.moveto(x=650, y=200, z=54)
            
            self._gripper_ctrl.open()     
            self._xarm_ctrl.moveto(x=650, y=200, z=100)
        def pile3(self):
            #pile3
                    
            self._xarm_ctrl.moveto(x=200, y=200, z=200)
            self._xarm_ctrl.moveto(x=500, y=200, z=200)
            self._xarm_ctrl.moveto(x=600, y=200, z=150)
            
            self._xarm_ctrl.moveto(x=656, y=200, z=95)
            
            self._gripper_ctrl.open()     
            self._xarm_ctrl.moveto(x=656, y=200, z=100)
        def pile4(self):
        #pile4
                
            self._xarm_ctrl.moveto(x=200, y=200, z=200)
            self._xarm_ctrl.moveto(x=500, y=200, z=200)
            self._xarm_ctrl.moveto(x=600, y=200, z=150)
            
            self._xarm_ctrl.moveto(x=656, y=200, z=138)
            
            self._gripper_ctrl.open()     
            self._xarm_ctrl.moveto(x=656, y=200, z=160)
        
        def errMsg():
            print ("_-_-_-_-_-_-_-_-_-_-_-_-_-_-_")
            print ("Command NOT understood")
            print ("(example)insert: r1 r2 b1 b2")
            print ("the order can be modified")
            print ("_-_-_-_-_-_-_-_-_-_-_-_-_-_-_")

        print ("Waiting for server:")
        #s=Color_Pos_client(a,b,c,d)
    
        if len(sys.argv)==5:
            a=str(sys.argv[1])
            b=str(sys.argv[2])
            c=str(sys.argv[3])
            d=str(sys.argv[4])

        else:
            print("No response")
            sys.exit(1)
        print ("Order received:")
        print ("|||||||||||||||")
        print (a,b,c,d)
        print ("|||||||||||||||")

        while True:

            self._gripper_ctrl.open()
            if not self._xarm_ctrl.moveto(z=self._safe_z):
                continue

#________________________________________________________________________________r1-case
#BEST CALIBRATION = r1 r2 b1 b2
            if a=='r1':
                cube1R(self)
                pile1(self)
                if b=='r2':
                    cube2R(self)
                    pile2(self)
                    if c=='b1':
                        cube1B(self)
                        pile3(self)
                        #__________b2
                        cube2B(self)
                        pile4(self)

                    elif c=='b2':
                        cube2B(self)
                        pile3(self)
                        #__________b1
                        cube1B(self)
                        pile4(self)

                    else:
                        errMsg()

                elif b=='b1':
                    cube1B(self)
                    pile2(self)
                    if c=='r2':
                        cube2R(self)
                        pile3(self)
                        #__________b2
                        cube2B(self)
                        pile4(self)

                    elif c=='b2':
                        cube2B(self)
                        pile3(self)
                        #__________r2
                        cube2R(self)
                        pile4(self)

                    else:
                        errMsg()

                elif b=='b2':
                    cube2B(self)
                    pile2(self)
                    if c=='r2':
                        cube2R(self)
                        pile3(self)
                        #__________b1
                        cube1B(self)
                        pile4(self)

                    elif c=='b1':
                        cube1B(self)
                        pile3(self)
                        #__________r2
                        cube2R(self)
                        pile4(self)

                    else:
                        errMsg()
                else:  
                    errMsg()
#________________________________________________________________________________r2-case

            elif a=='r2':
                cube2R(self)
                pile1(self)
                if b=='r1':
                    cube1R(self)
                    pile2(self)
                    if c=='b1':
                        cube1B(self)
                        pile3(self)
                        #__________b2
                        cube2B(self)
                        pile4(self)

                    elif c=='b2':
                        cube2B(self)
                        pile3(self)
                        #__________b1
                        cube1B(self)
                        pile4(self)

                    else:
                        errMsg()

                elif b=='b1':
                    cube1B(self)
                    pile2(self)
                    if c=='r1':
                        cube1R(self)
                        pile3(self)
                        #__________b2
                        cube2B(self)
                        pile4(self)

                    elif c=='b2':
                        cube2B(self)
                        pile3(self)
                        #__________r1
                        cube1R(self)
                        pile4(self)

                    else:
                        errMsg()

                elif b=='b2':
                    cube2B(self)
                    pile2(self)
                    if c=='r1':
                        cube1R(self)
                        pile3(self)
                        #__________b1
                        cube1B(self)
                        pile4(self)

                    elif c=='b1':
                        cube1B(self)
                        pile3(self)
                        #__________r1
                        cube1R(self)
                        pile4(self)

                    else:
                        errMsg()
                else:  
                    errMsg()

#________________________________________________________________________________b1-case

            elif a=='b1':
                cube1B(self)
                pile1(self)
                if b=='r1':
                    cube1R(self)
                    pile2(self)
                    if c=='r2':
                        cube2R(self)
                        pile3(self)
                        #__________b2
                        cube2B(self)
                        pile4(self)

                    elif c=='b2':
                        cube2B(self)
                        pile3(self)
                        #__________r2
                        cube2R(self)
                        pile4(self)

                    else:
                        errMsg()

                elif b=='b2':
                    cube2B(self)
                    pile2(self)
                    if c=='r1':
                        cube1R(self)
                        pile3(self)
                        #__________r2
                        cube2R(self)
                        pile4(self)

                    elif c=='r2':
                        cube2R(self)
                        pile3(self)
                        #__________r1
                        cube1R(self)
                        pile4(self)

                    else:
                        errMsg()

                elif b=='r2':
                    cube2R(self)
                    pile2(self)
                    if c=='r1':
                        cube1R(self)
                        pile3(self)
                        #__________b2
                        cube2B(self)
                        pile4(self)

                    elif c=='b2':
                        cube2B(self)
                        pile3(self)
                        #__________r1
                        cube1R(self)
                        pile4(self)

                    else:
                        errMsg()
                else:  
                    errMsg()

#________________________________________________________________________________r2-case

            elif a=='b2':
                cube2B(self)
                pile1(self)
                if b=='r1':
                    cube1R(self)
                    pile2(self)
                    if c=='r2':
                        cube2R(self)
                        pile3(self)
                        #__________b1
                        cube1B(self)
                        pile4(self)

                    elif c=='b1':
                        cube1B(self)
                        pile3(self)
                        #__________r2
                        cube2R(self)
                        pile4(self)

                    else:
                        errMsg()

                elif b=='b1':
                    cube1B(self)
                    pile2(self)
                    if c=='r1':
                        cube1R(self)
                        pile3(self)
                        #__________r2
                        cube2R(self)
                        pile4(self)

                    elif c=='r2':
                        cube2R(self)
                        pile3(self)
                        #__________r1
                        cube1R(self)
                        pile4(self)

                    else:
                        errMsg()

                elif b=='r2':
                    cube2R(self)
                    pile2(self)
                    if c=='r1':
                        cube1R(self)
                        pile3(self)
                        #__________b1
                        cube1B(self)
                        pile4(self)

                    elif c=='b1':
                        cube1B(self)
                        pile3(self)
                        #__________r1
                        cube1R(self)
                        pile4(self)

                    else:
                        errMsg()
                else:  
                    errMsg()
#________________________________________________________________________________FAIL-case_1            
            else:
                errMsg()
            
            


            self._xarm_ctrl.moveto(x=200, y=200, z=200)
            print ("<><><><><><><>")
            print ("End of routine")
            print ("<><><><><><><>")
            break
        
#__________________________________________________________________

class GazeboCamera(object):
    def __init__(self, topic_name='/camera/image_raw/compressed'):
        self._frame_que = queue.Queue(10)
        self._bridge = CvBridge()
        self._img_sub = rospy.Subscriber(topic_name, CompressedImage, self._img_callback)

    def _img_callback(self, data):
        if self._frame_que.full():
            self._frame_que.get()
        self._frame_que.put(self._bridge.compressed_imgmsg_to_cv2(data))
    
    def get_frame(self):
        if self._frame_que.empty():
            return None
        return self._frame_que.get()
#__________________________________________________________________

def get_recognition_rect(frame, lower=COLOR_DICT['red']['lower'], upper=COLOR_DICT['red']['upper'], show=True):
    gs_frame = cv2.GaussianBlur(frame, (5, 5), 0)
    hsv = cv2.cvtColor(gs_frame, cv2.COLOR_BGR2HSV)
    erode_hsv = cv2.erode(hsv, None, iterations=2)
    cv2.imshow("erode", erode_hsv)
    inRange_hsv = cv2.inRange(erode_hsv, lower, upper)
    cv2.imshow("in_hsv", inRange_hsv)
    contours = cv2.findContours(inRange_hsv.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    rects = []
    for _, c in enumerate(contours):
        rect = cv2.minAreaRect(c)
        if rect[1][0] < 20 or rect[1][1] < 20:
            continue
        # print(rect)
        if PY3:
            box = cv2.boxPoints(rect)
            
        else:
            box = cv2.cv.BoxPoints(rect)
        cv2.drawContours(frame, [np.int0(box)], -1, (0, 255, 255), 1)
        rects.append(rect)
    
    if show:
        cv2.imshow("Frame", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            rospy.signal_shutdown('key to exit')
    return rects
#__________________________________________________________________
def Color_Pos_client(a,b,c,d):
    rospy.wait_for_service('Color_Pos')
    try:
        Color_Pos=rospy.ServiceProxy('Color_Pos',ColorPosition)
        respC=Color_Pos(a,b,c,d)
        return respC.af, respC.bs, respC.cs, respC.ds
    except rospy.ServiceException:
        print ("Service call failed")
#__________________________________________________________________
if __name__ == '__main__':
    rospy.init_node('color_recognition_node', anonymous=False)
    dof = rospy.get_param('/xarm/DOF', default=6)
    rate = rospy.Rate(10.0)

    motion_que = queue.Queue(1)
    motion = GazeboMotionThread(motion_que, dof=dof)
    motion.start()

    color = COLOR_DICT['red']

    cam = GazeboCamera(topic_name='/camera/image_raw/compressed')

    while not rospy.is_shutdown():
        rate.sleep()
        frame = cam.get_frame()
        if frame is None:
            continue
        rects = get_recognition_rect(frame, lower=color['lower'], upper=color['upper'])
        if len(rects) == 0:
            continue
        if motion.in_motion or motion_que.qsize() != 0:
            continue
        motion_que.put(rects)
