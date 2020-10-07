import rospy
from gazebo_msgs.msg import ModelStates

class model_stateee():
    def __init__(self):
        rospy.init_node("try",anonymous=False)
        rospy.Subscriber("/gazebo/model_states",ModelStates,self.data)
        print("aaa")
        # data()
    def Gazebo_aa_box_state(self):
        
        
        pass


    def data(self,data):
        # print("sssssssssssssss")
        # try:
        self.aa_box_pos_x = data.pose[6].position.x
        print(self.aa_box_pos_x)
            
        # except CvBridgeError as e:
        #     print(e)

if __name__ == "__main__":
    a = model_stateee()