#!/usr/bin/env python

import gym
import numpy as np
import time
import qlearn
from gym import wrappers
# ROS packages required
import rospy
import rospkg
from openai_ros.openai_ros_common import StartOpenAI_ROS_Environment
from imagetranformer import ImageTransformer
from rl_common import ReplayMemory, update_state, learn
from dqn_model import DQN
import cv2
import tensorflow as tf
from datetime import datetime
import sys

MAX_EXPERIENCE = 50000
MIN_EXPERIENCE = 100
TARGET_UPDATE_PERIOD = 10000
IM_SIZE = 84
K = 3
n_history = 4


def play_ones(env,
              sess,
              total_t,
              experience_replay_buffer,
              model,
              target_model,
              image_transformer,
              gamma,
              batch_size,
              epsilon,
              epsilon_change,
              epsilon_min,
              pathOut,
              record):
    
    t0 = datetime.now()
    obs = env.reset()
    
    obs_small = image_transformer.transform(obs, sess)
    state = np.stack([obs_small] * n_history, axis = 2)
    loss = None
    
    total_time_training = 0
    num_steps_in_episode = 0
    episode_reward = 0
    record = True
    done = False
    if record == True:
        #out = cv2.VideoWriter(pathOut,cv2.VideoWriter_fourcc(*'XVID'), 20.0, (480,640))
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter('output.avi',fourcc, 20.0, (640,480))
    while not done:
        
        if total_t % TARGET_UPDATE_PERIOD == 0:
            target_model.copy_from(model)
            print("model is been copied!")
        
        action = model.sample_action(state, epsilon)
        obs, reward, done, _ = env.step(action)
        obs_small = image_transformer.transform(obs, sess)
        next_state = update_state(state, obs_small)
        
        episode_reward += reward
        
        experience_replay_buffer.add_experience(action, obs_small, reward, done)
        t0_2 = datetime.now()
        loss = learn(model, target_model, experience_replay_buffer, gamma, batch_size)
        dt = datetime.now() - t0_2
        
        total_time_training += dt.total_seconds()
        num_steps_in_episode += 1
        
        state = next_state
        total_t += 1
        epsilon = max(epsilon - epsilon_change, epsilon_min)
        if record == True:
            frame = cv2.resize(obs,(640,480))
            out.write(frame)
            #cv2.imshow("frame", frame)
    if record == True:
        out.release()
    return total_t, episode_reward, (datetime.now()-t0), num_steps_in_episode, total_time_training/num_steps_in_episode, loss


if __name__ == '__main__':
    print "Starting training!!!"
    conv_layer_sizes = [(32,8,4), (64,4,2), (64,3,1)]
    hidden_layer_sizes = [512]
    gamma = 0.99
    batch_sz = 32
    num_episodes = 3500
    total_t = 0
    experience_replay_buffer = ReplayMemory()
    episode_rewards = np.zeros(num_episodes)
    episode_lens = np.zeros(num_episodes)
    
    epsilon = 1.0
    epsilon_min = 0.1
    epsilon_change = (epsilon - epsilon_min) / 500000

    rospy.init_node('sumo_dqlearn',
                    anonymous=True, log_level=rospy.WARN)

    # Init OpenAI_ROS ENV
    task_and_robot_environment_name = rospy.get_param(
        '/turtlebot2/task_and_robot_environment_name')
    env = StartOpenAI_ROS_Environment(
        task_and_robot_environment_name)
    rospy.loginfo("Gym environment done")   
    obs = env.reset()
    print obs.shape
    
    # Create the Gym environment
    rospy.loginfo("Gym environment done")
    rospy.loginfo("Starting Learning")

    # Set the logging system
    rospack = rospkg.RosPack()
    pkg_path = rospack.get_path('gazebo_sumo')
    #outdir = pkg_path + '/training_results'
    #env = wrappers.Monitor(env, outdir, force=True)
    rospy.loginfo("Monitor Wrapper started")

    last_time_steps = np.ndarray(0)

    # Loads parameters from the ROS param server
    # Parameters are stored in a yaml file inside the config directory
    # They are loaded at runtime by the launch file
    Alpha = rospy.get_param("/turtlebot2/alpha")
    Epsilon = rospy.get_param("/turtlebot2/epsilon")
    Gamma = rospy.get_param("/turtlebot2/gamma")
    epsilon_discount = rospy.get_param("/turtlebot2/epsilon_discount")
    nepisodes = rospy.get_param("/turtlebot2/nepisodes")
    nsteps = rospy.get_param("/turtlebot2/nsteps")

    running_step = rospy.get_param("/turtlebot2/running_step")

    # Initialises the algorithm that we are going to use for learning
    #qlearn = qlearn.QLearn(actions=range(env.action_space.n),
     #                      alpha=Alpha, gamma=Gamma, epsilon=Epsilon)
    #initial_epsilon = qlearn.epsilon

    start_time = time.time()
    highest_reward = 0

    model = DQN(
            K = K,
            conv_layer_sizes=conv_layer_sizes,
            hidden_layer_sizes=hidden_layer_sizes,
            scope="model",
            image_size=IM_SIZE
            )
    
    target_model = DQN(
            K = K,
            conv_layer_sizes=conv_layer_sizes,
            hidden_layer_sizes=hidden_layer_sizes,
            scope="target_model",
            image_size=IM_SIZE
            )
    
    image_transformer = ImageTransformer(IM_SIZE)

    

    
    with tf.Session() as sess:
        model.set_session(sess)
        target_model.set_session(sess)
        #model.load()
        #target_model.load()
        sess.run(tf.global_variables_initializer())
        print("Initializing experience replay buffer...")
        obs = env.reset()
        
        for i in range(MIN_EXPERIENCE):
            action = np.random.choice(K)
            obs, reward, done, _ = env.step(action)
            obs_small = image_transformer.transform(obs, sess)
            experience_replay_buffer.add_experience(action, obs_small, reward, done)
            if done:
                obs = env.reset()

        
        print("Done! Starts Training!!")     
        t0 = datetime.now()
        record = True
        for i in range(num_episodes):
            
            total_t, episode_reward, duration, num_steps_in_episode, time_per_step, epsilon = play_ones(
                    env,
                    sess,
                    total_t,
                    experience_replay_buffer,
                    model,
                    target_model,
                    image_transformer,
                    gamma,
                    batch_sz,
                    epsilon,
                    epsilon_change,
                    epsilon_min,
                    None,
                    False)
            episode_rewards[i] = episode_reward
            episode_lens[i] = num_steps_in_episode
            last_100_avg = episode_rewards[max(0,i-100):i+1].mean()
            print("Episode:", i ,
                  "Duration:", duration,
                  "Num steps:", num_steps_in_episode,
                  "Reward:", episode_reward,
                  "Training time per step:", "%.3f" %time_per_step,
                  "Avg Reward:", "%.3f"%last_100_avg,
                  "Epsilon:", "%.3f"%epsilon)
            sys.stdout.flush()
        print("Total duration:", datetime.now()-t0)
        model.save()
        
        y = smooth(episode_rewards)
        plt.plot(episode_rewards, label='orig')
        plt.plot(y, label='smoothed')
        plt.legend()
        plt.show()
        env.close()
        

'''
    # Starts the main training loop: the one about the episodes to do
    for x in range(nepisodes):
        rospy.logdebug("############### WALL START EPISODE=>" + str(x))

        cumulated_reward = 0
        done = False
       # if qlearn.epsilon > 0.05:
        #    qlearn.epsilon *= epsilon_discount

        # Initialize the environment and get first state of the robot
        observation = env.reset()
        state = ''.join(map(str, observation))

        # Show on screen the actual situation of the robot
        # env.render()
        # for each episode, we test the robot for nsteps
        for i in range(nsteps):
            rospy.logwarn("############### Start Step=>" + str(i))
            # Pick an action based on the current state
            #action = qlearn.chooseAction(state)
	    action = [0]
            rospy.logwarn("Next action is:%d", action)
            # Execute the action in the environment and get feedback
            observation, reward, done, info = env.step(action)

            rospy.logwarn(str(observation) + " " + str(reward))
            cumulated_reward += reward
            if highest_reward < cumulated_reward:
                highest_reward = cumulated_reward

            nextState = ''.join(map(str, observation))

            # Make the algorithm learn based on the results
            rospy.logwarn("# state we were=>" + str(state))
            rospy.logwarn("# action that we took=>" + str(action))
            rospy.logwarn("# reward that action gave=>" + str(reward))
            rospy.logwarn("# episode cumulated_reward=>" +
                          str(cumulated_reward))
            rospy.logwarn(
                "# State in which we will start next step=>" + str(nextState))
            #qlearn.learn(state, action, reward, nextState)

            if not (done):
                rospy.logwarn("NOT DONE")
                state = nextState
            else:
                rospy.logwarn("DONE")
                last_time_steps = numpy.append(last_time_steps, [int(i + 1)])
                break
            rospy.logwarn("############### END Step=>" + str(i))
            #raw_input("Next Step...PRESS KEY")
            # rospy.sleep(2.0)
        m, s = divmod(int(time.time() - start_time), 60)
        h, m = divmod(m, 60)
        #rospy.logerr(("EP: " + str(x + 1) + " - [alpha: " + str(round(qlearn.alpha, 2)) + " - gamma: " + str(
        #    round(qlearn.gamma, 2)) + " - epsilon: " + str(round(qlearn.epsilon, 2)) + "] - Reward: " + str(
        #    cumulated_reward) + "     Time: %d:%02d:%02d" % (h, m, s)))

    #rospy.loginfo(("\n|" + str(nepisodes) + "|" + str(qlearn.alpha) + "|" + str(qlearn.gamma) + "|" + str(
     #   initial_epsilon) + "*" + str(epsilon_discount) + "|" + str(highest_reward) + "| PICTURE |"))

    l = last_time_steps.tolist()
    l.sort()

    # print("Parameters: a="+str)
    rospy.loginfo("Overall score: {:0.2f}".format(last_time_steps.mean()))
    rospy.loginfo("Best 100 score: {:0.2f}".format(
        reduce(lambda x, y: x + y, l[-100:]) / len(l[-100:])))

    env.close()
    '''



