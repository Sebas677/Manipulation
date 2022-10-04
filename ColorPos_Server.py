#!/usr/bin/env python


from xarm_msgs.srv import ColorPosition
from xarm_msgs.srv import ColorPositionRequest
from xarm_msgs.srv import ColorPositionResponse

import rospy

#__________________________________________________________________
def handle_Color_Pos(req):
    print ("Returning [%s %s %s %s]" %(req.a, req.b, req.c, req.d))
    print (req)
    return ColorPositionResponse(req.a, req.b, req.c, req.d)

def Color_Pos_server():
    rospy.init_node('Color_Pos_server')
    s=rospy.service('Color_Pos', ColorPosition, handle_Color_Pos)
    print("Insert 4 color initials(r-b)+num(1-2) with a space between each (b1 r2 r1 b2)")
    rospy.spin()

if __name__ == '__main__':
    Color_Pos_server()
