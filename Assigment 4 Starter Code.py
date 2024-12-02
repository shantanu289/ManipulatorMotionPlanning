import pybullet as p
import pybullet_data
import time
from useful_code import *
import random
from matplotlib import pyplot as plt   
import numpy as np
from scipy.spatial import KDTree


def edge_cost(node1_loc, node2_loc):
    # node1_loc -> np.array
    # node2_loc -> np.array
    return np.linalg.norm(node2_loc-node1_loc)

def check_node_collision(robot_id, object_ids, joint_position):
    """
    Checks for collisions between a robot and an object in PyBullet. 

    Args:
        robot_id (int): The ID of the robot in PyBullet.
        object_id (int): The ID of the object in PyBullet.
        joint_position (list): List of joint positions. 

    Returns:
        bool: True if a collision is detected, False otherwise.
    """
    # set joint positions
    for joint_index, joint_pos in enumerate(joint_position):
        p.resetJointState(robot_id, joint_index, joint_pos)

    # Perform collision check for all links
    for object_id in object_ids:    # Check for each object
        for link_index in range(0, p.getNumJoints(robot_id)): # Check for each link of the robot
            contact_points = p.getClosestPoints(
                bodyA=robot_id, bodyB=object_id, distance=0.01, linkIndexA=link_index
            )
            if contact_points:  # If any contact points exist, a collision is detected
                return True # exit early
    return False

#################################################
#### YOUR CODE HERE: COLLISION EDGE CHECKING ####
#################################################
def check_edge_collision(robot_id, object_ids, joint_position_start, joint_position_end, discretization_step=0.01):
    """ 
    Checks for collision between two joint positions of a robot in PyBullet.
    Args:
        robot_id (int): The ID of the robot in PyBullet.
        object_ids (list): List of IDs of the objects in PyBullet.
        joint_position_start (list): List of joint positions to start from.
        joint_position_end (list): List of joint positions to get to.
        discretization_step (float): maximum interpolation distance before a new collision check is performed.
    Returns:
        bool: True if a collision is detected, False otherwise.
    """
    joint_start_np = np.array(joint_position_start)
    joint_end_np = np.array(joint_position_end)
    num_points = int(np.linalg.norm(joint_end_np - joint_start_np, np.inf) / discretization_step)   

    joint_angle_list = [joint_position_start]
    for i in range(1, num_points):
        joint_angle_intermediate = joint_start_np + i*(joint_end_np - joint_start_np)/num_points
        joint_angle_list.append(list(joint_angle_intermediate))
    joint_angle_list.append(list(joint_position_end))
    collision_flag = 0
    for joint_ in joint_angle_list:
        if check_node_collision(robot_id, object_ids, joint_):
            collision_flag = 1
            break
    if (collision_flag == 1):
        return True
    else :
        return False

    
# Provided 
class Node:
    def __init__(self, joint_angles):
        self.joint_angles = np.array(joint_angles)  # joint angles of the node in n-dimensional space
        self.parent = None
        self.cost = 0

######################################################################
##################### YOUR CODE HERE: RRT CLASS ######################
######################################################################
class RRT:
    def __init__(self, q_start, q_goal, robot_id, obstacle_ids, q_limits, max_iter=10000, step_size=0.5):
        """
        RRT Initialization.

        Parameters:
        - q_start: List of starting joint angles [x1, x2, ..., xn].
        - q_goal: List of goal joint angles [x1, x2, ..., xn].
        - obstacle_ids: List of obstacles, each as a tuple ([center1, center2, ..., centern], radius).
        - q_limits: List of tuples [(min_x1, max_x1), ..., (min_xn, max_xn)] representing the limits in each dimension.
        - max_iter: Maximum number of iterations.
        - step_size: Maximum step size to expand the tree.
        - node_list : List of NODE objects created in the tree
        - node_pos_list : List of NODE locations of nodes in the tree (1-1 correspondence with node_list)
        - q_limits_np : numpy array of limits
        - pathToGoal : list [] of locations (np.array) of the path from start to goal
        - R : radius to update parent based on cost in RRT*
        """
        self.q_start = Node(q_start)
        self.q_goal = Node(q_goal)
        self.obstacle_ids = obstacle_ids
        self.robot_id = robot_id
        self.q_limits = q_limits
        self.max_iter = max_iter
        self.step_size = step_size
        self.node_list = [self.q_start]
        self.node_pos_list = [self.q_start.joint_angles]
        self.q_limits_np = np.array(self.q_limits)
        self.pathToGoal = []
        self.R = 1

    def step(self, from_node, to_joint_angles):
        """Step from "from_node" to "to_joint_angles", that should
         (a) return the to_joint_angles if it is within the self.step_size or
         (b) only step so far as self.step_size, returning the new node within that distance"""
        # "to_joint_angles" --> np.array()
        # the ONLY check performed before coming inside this function is that "to_joint_angles" is NOT itself on an obstacle
        dist_ = np.linalg.norm(to_joint_angles - from_node.joint_angles, np.inf)
        
        if (dist_ <= self.step_size):
            if (check_edge_collision(self.robot_id, self.obstacle_ids, list(from_node.joint_angles), list(to_joint_angles)) == False):
                new_node = Node(to_joint_angles)
                new_node.parent = from_node
                self.node_list.append(new_node)
                self.node_pos_list.append(to_joint_angles)                            
        else :
            new_joint_angles = from_node.joint_angles + self.step_size*(to_joint_angles - from_node.joint_angles)/dist_
            if (check_edge_collision(self.robot_id, self.obstacle_ids, list(from_node.joint_angles), list(new_joint_angles)) == False):
                new_node = Node(new_joint_angles)
                new_node.parent = from_node
                self.node_list.append(new_node)
                self.node_pos_list.append(new_joint_angles)
                
    
    def get_nearest_node(self, random_point):
        """Find the nearest node in the tree to a given point."""
        # "random_point" --> np.array()
        # Before coming inside this function, it has been checked that the "random_point" is NOT on an obstacle
        tree = KDTree(np.array(self.node_pos_list))
        dist_, ind_ = tree.query(random_point, k=1)
        return self.node_list[ind_]

    def plan(self):
        """Run the RRT algorithm to find a path of dimension Nx3. Limit the search to only max_iter iterations."""

        goal_reached = 0
        for i in range(self.max_iter):
            ## select a random point within the q_limit range
            x = np.random.rand()
            if (x > 0.2):
                random_point = np.random.rand(self.q_limits_np.shape[0])*(self.q_limits_np[:,1] - self.q_limits_np[:,0]) + self.q_limits_np[:,0]                 
            else :
                random_point = self.q_goal.joint_angles  

            ## check that the point sampled is NOT on an obstacle - if it is : New point, else : perform the next tasks            
            if (check_node_collision(self.robot_id, self.obstacle_ids, list(random_point)) == True):
                continue

            ## get the nearest node from node_list to the sampled point location
            nearest_node = self.get_nearest_node(random_point)
            
            ## step from the "nearest node" to the "sampled location" - if stepping involves edge collision : NO new node added to list, else : added
            self.step(nearest_node, random_point)
            ## check if the latest node added can be directly connected to the goal without collision            
            if (np.linalg.norm(self.q_goal.joint_angles - self.node_list[-1].joint_angles) < 0.1) and (check_edge_collision(self.robot_id, self.obstacle_ids, list(self.node_list[-1].joint_angles), list(self.q_goal.joint_angles))==False):
                self.q_goal.parent = self.node_list[-1]
                goal_reached = 1
                break
            

        if (goal_reached == 1):
            # Populates the path to the RRT object
            self.getPath()
        else:
            print("Nodes list length", len(self.node_list))
            print("Goal not reached even after MAX_ITERATIONS")

    def getPath(self):
        # returns a list of np.array locations from start to goal
        path_joint_angle_list = []
        curr_node = self.q_goal
        while (curr_node.parent != None):
            path_joint_angle_list.append(curr_node.joint_angles)
            curr_node = curr_node.parent
        path_joint_angle_list.append(self.q_start.joint_angles)
        self.pathToGoal = path_joint_angle_list
        self.pathToGoal.reverse()

    def getNearestNeighbors(self, latest_node):
        # latest_node -> NODE object corresponding to the latest added node. This MUST be = self.node_list[-1]
        nn_list = []
        for j in range(len(self.node_list)-1):
            if (edge_cost(latest_node.joint_angles, self.node_list[j].joint_angles) <= self.R) and (check_edge_collision(self.robot_id, self.obstacle_ids, list(self.node_list[j].joint_angles), list(latest_node.joint_angles)) == False):
                nn_list.append(j)
        return nn_list

    def plan2(self):

        goal_reached = 0
        for i in range(self.max_iter):

            # sample a random point in the c-space  
            x = np.random.rand()
            if (x > 0.2):
                random_point = np.random.rand(self.q_limits_np.shape[0])*(self.q_limits_np[:,1] - self.q_limits_np[:,0]) + self.q_limits_np[:,0]                 
            else :
                random_point = self.q_goal.joint_angles
            
            # check if the sample point is ON an obstacle. If so - continue
            if (check_node_collision(self.robot_id, self.obstacle_ids, list(random_point)) == True):
                continue

            # get the nearest node
            nearest_node = self.get_nearest_node(random_point)
            node_list_len = len(self.node_list)

            # step to the nearest node and form a new node at a certain location. Assign a parent, cost, add node to the list
            self.step(nearest_node, random_point)
            if (len(self.node_list) == node_list_len):
                # no new node added to the list
                continue

            # update the cost of the node recently added
            latest_node = self.node_list[-1]
            latest_node.cost = latest_node.parent.cost + edge_cost(latest_node.joint_angles, latest_node.parent.joint_angles)

            # get the list of nearest nodes in a radius (R=1) to the latest node (without collision) - if no node in this area then : continue
            # list_of_nearest_neighbors = list of INDICES of nodes from node_list which are within radius
            list_of_nearest_neighbors = self.getNearestNeighbors(latest_node)
            if (len(list_of_nearest_neighbors) == 1):
                continue

            # update the cost of the latest_node based on these neighbouring nodes
            for nn in list_of_nearest_neighbors:
                nearest_n = self.node_list[nn]
                if (nearest_n.cost + edge_cost(nearest_n.joint_angles, latest_node.joint_angles) < latest_node.cost):
                    latest_node.parent = nearest_n
                    latest_node.cost = nearest_n.cost + edge_cost(nearest_n.joint_angles, latest_node.joint_angles)

            # based on the updated cost of the "latest_node" update the cost of each of the neighbouring nodes if path through "latest_node" is better
            for n in list_of_nearest_neighbors:
                nearest_neighbor = self.node_list[n]
                if (latest_node.cost + edge_cost(latest_node.joint_angles, nearest_neighbor.joint_angles) < nearest_neighbor.cost):
                    nearest_neighbor.parent = latest_node
                    nearest_neighbor.cost = latest_node.cost + edge_cost(latest_node.joint_angles, nearest_neighbor.joint_angles)
                    
            # having done the parent, cost updates, check if we're near the goal position
            if (np.linalg.norm(self.q_goal.joint_angles - self.node_list[-1].joint_angles) < 0.1) and (check_edge_collision(self.robot_id, self.obstacle_ids, list(self.node_list[-1].joint_angles), list(self.q_goal.joint_angles))==False):
                self.q_goal.parent = self.node_list[-1]
                self.q_goal.cost = self.node_list[-1].cost + edge_cost(self.node_list[-1].joint_angles, self.q_goal.joint_angles)
                goal_reached = 1
                break
        
        if (goal_reached == 1):
            # Populates the path to the RRT object
            self.getPath()
        else:
            print("Nodes list length", len(self.node_list))
            print("Goal not reached even after MAX_ITERATIONS")


#####################################################
##################### MAIN CODE #####################
#####################################################

if __name__ == "__main__":
    
    #######################
    #### PROBLEM SETUP ####
    #######################

    # Initialize PyBullet
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath()) # For default URDFs
    p.setGravity(0, 0, -9.8)

    # Load the plane and robot arm
    ground_id = p.loadURDF("plane.urdf")
    arm_id = p.loadURDF("three_link_arm.urdf", [0, 0, 0], useFixedBase=True)

    # Add Collision Objects
    collision_ids = [ground_id] # add the ground to the collision list
    collision_positions = [[0.3, 0.5, 0.251], [-0.3, 0.3, 0.101], [-1, -0.15, 0.251], [-1, -0.15, 0.752], [-0.5, -1, 0.251], [0.5, -0.35, 0.201], [0.5, -0.35, 0.602]]
    collision_orientations =  [[0, 0, 0.5], [0, 0, 0.2], [0, 0, 0],[0, 0, 1], [0, 0, 0], [0, 0, .25], [0, 0, 0.5]]
    collision_scales = [0.5, 0.25, 0.5, 0.5, 0.5, 0.4, 0.4]
    for i in range(len(collision_scales)):
        collision_ids.append(p.loadURDF("cube.urdf",
            basePosition=collision_positions[i],  # Position of the cube
            baseOrientation=p.getQuaternionFromEuler(collision_orientations[i]),  # Orientation of the cube
            globalScaling=collision_scales[i]  # Scale the cube to half size
        ))

    # Goal Joint Positions for the Robot
    goal_positions = [[-2.54, 0.15, -0.15], [-1.82,0.15,-0.15],[0.5, 0.15,-0.15], [1.7,0.2,-0.15],[-2.54, 0.15, -0.15]]

    # Joint Limits of the Robot
    joint_limits = [[-np.pi, np.pi], [0, np.pi], [-np.pi, np.pi]]

    # A3xN path array that will be filled with waypoints through all the goal positions
    path_saved = [np.array([[-2.54, 0.15, -0.15]])] # Start at the first goal position

    ####################################################################################################
    #### YOUR CODE HERE: RUN RRT MOTION PLANNER FOR ALL goal_positions (starting at goal position 1) ###
    ####################################################################################################

    for g in range(1, len(goal_positions)):
           rrt = RRT(goal_positions[g-1], goal_positions[g], arm_id, collision_ids, joint_limits)
           rrt.plan()
           path_to_goal = rrt.pathToGoal
           path_saved = path_saved + path_to_goal
           print("Path from ", goal_positions[g-1], " to ", goal_positions[g], " : ", len(path_to_goal))


    ################################################################################
    ####  RUN THE SIMULATION AND MOVE THE ROBOT ALONG PATH_SAVED ###################
    ################################################################################

    # Set the initial joint positions
    for joint_index, joint_pos in enumerate(goal_positions[0]):
        p.resetJointState(arm_id, joint_index, joint_pos)

    # Move through the waypoints
    for waypoint in path_saved:
        # "move" to next waypoints
        for joint_index, joint_pos in enumerate(waypoint):
        # run velocity control until waypoint is reached
            while True:
                #get current joint positions
                goal_positions = [p.getJointState(arm_id, i)[0] for i in range(3)]
                # calculate the displacement to reach the next waypoint
                displacement_to_waypoint = waypoint-goal_positions
                # check if goal is reached
                max_speed = 0.05
                if(np.linalg.norm(displacement_to_waypoint) < max_speed):
                    break
                else:
                    # calculate the "velocity" to reach the next waypoint
                    velocities = np.min((np.linalg.norm(displacement_to_waypoint), max_speed))*displacement_to_waypoint/np.linalg.norm(displacement_to_waypoint)
                    for joint_index, joint_step in enumerate(velocities):
                        p.setJointMotorControl2(
                            bodyIndex=arm_id,
                            jointIndex=joint_index,
                            controlMode=p.VELOCITY_CONTROL,
                            targetVelocity=joint_step,
                        )
                        
                #Take a simulation step
                p.stepSimulation()            
        time.sleep(1.0 / 240.0)


    # Disconnect from PyBullet
    time.sleep(100) # Remove this line -- it is just to keep the GUI open when you first run this starter code
    p.disconnect()