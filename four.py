import glob
import os
import sys
import random
import time
import numpy as np
import cv2
import math
import settings
from collections import deque
import queue
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.layers import Dense, Input, Activation, Flatten, Reshape
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import AveragePooling1D
from tensorflow.keras.layers import concatenate
from dataclasses import dataclass

import tensorflow as tf
#import keras.backend.tensorflow_backend as backend
#from threading import Thread

from tqdm import tqdm
from os import system

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass
import carla

tf.compat.v1.disable_eager_execution()

os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
#physical_devices = tf.config.list_physical_devices('GPU')
#tf.config.experimental.set_memory_growth(physical_devices[0], True)
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


SHOW_PREVIEW = True
IM_WIDTH = 640
IM_HEIGHT = 480
SECONDS_PER_EPISODE = 300
REPLAY_MEMORY_SIZE = 5_000
MIN_REPLAY_MEMORY_SIZE = 30
MINIBATCH_SIZE = 30
PREDICTION_BATCH_SIZE = 1
TRAINING_BATCH_SIZE = MINIBATCH_SIZE // 4
UPDATE_TARGET_EVERY = 5
MODEL_NAME = "Actor_Critic"
NUM_ACTIONS = 9

MEMORY_FRACTION = 0.4
MIN_REWARD = -200

EPISODES = 1

DISCOUNT = 0.99
epsilon = 0.69
EPSILON_DECAY = 0.998 ## 0.9975 99975
MIN_EPSILON = 0.2
GAMMA = 0.99

AGGREGATE_STATS_EVERY = 2


@dataclass
class ACTIONS:
    forward = 0
    left = 1
    right = 2
    forward_left = 3
    forward_right = 4
    brake = 5
    brake_left = 6
    brake_right = 7
    no_action = 8

ACTION_CONTROL = {
        0: [1, 0, 0],
        1: [0, 0, -1],
        2: [0, 0, 1],
        3: [1, 0 ,-1],
        4: [1, 0 ,1],
        5: [0, 1, 0],
        6: [0, 1, -1],
        7: [0, 1, 1],
        8: None
    }

ACTION_NAMES = {
        0: 'forward',
        1: 'left',
        2: 'right',
        3: 'forward_left',
        4: 'forward_right',
        5: 'brake',
        6: 'brake_left',
        7: 'brake_right',
        8: 'no_action'
    }




def clear():
    _ = system('cls')

# Own Tensorboard class
class ModifiedTensorBoard(TensorBoard):

    # Overriding init to set initial step and writer (we want one log file for all .fit() calls)
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.step = 1
        self.writer = tf.compat.v1.summary.FileWriter('./logs')

    # Overriding this method to stop creating default log writer
    def set_model(self, model):
        pass

    # Overrided, saves logs with our step number
    # (otherwise every .fit() will start writing from 0th step)
    def on_epoch_end(self, epoch, logs=None):
        self.update_stats(**logs)

    # Overrided
    # We train for one batch only, no need to save anything at epoch end
    def on_batch_end(self, batch, logs=None):
        pass

    # Overrided, so won't close writer
    def on_train_end(self, _):
        pass

    # Custom method for saving own metrics
    # Creates writer, writes custom metrics and closes writer
    def update_stats(self, **stats):
        self._write_logs(stats, self.step)

    def _write_logs(self, logs, index):
        for name, value in logs.items():
            if name in ['batch', 'size']:
                continue
            summary = tf.compat.v1.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = value
            summary_value.tag = name
            self.writer.add_summary(summary, index)
        self.writer.flush()


class CarEnv:
    SHOW_CAM = SHOW_PREVIEW
    STEER_AMT = 1.0
    im_width = IM_WIDTH
    im_height = IM_HEIGHT
    front_camera = None
    MAKE_VID = False
    episode_start = None

    def __init__(self):
        self.client = carla.Client("localhost", 2000)
        self.client.set_timeout(20.0)
        self.world = self.client.get_world()
        self.blueprint_library = self.world.get_blueprint_library()
        self.model_3 = self.blueprint_library.filter("model3")[0]
        self.img_queue = queue.Queue()
        self.lidar_queue = queue.Queue()
        self.radar_queue = queue.Queue()
        self.actions = [getattr(ACTIONS, action) for action in settings.ACTIONS]
        self.video_writer = cv2.VideoWriter('Day_0.avi', cv2.VideoWriter_fourcc(*'DIVX'),5,(640,480))
        self.world.tick()

    def reset(self):
        #print("reset called")
        self.collision_hist = []
        self.actor_list = []

        #self.transform = random.choice(self.world.get_map().get_spawn_points())
        #self.vehicle = self.world.spawn_actor(self.model_3, self.transform)

        spawn_start = time.time()
        while True:
            try:
                # Get random spot from a list from predefined spots and try to spawn a car there
                self.transform = random.choice(self.world.get_map().get_spawn_points())
                self.vehicle = self.world.spawn_actor(self.model_3, self.transform)
                break
            except:
                time.sleep(0.01)

            # If that can't be done in 3 seconds - forgive (and allow main process to handle for this problem)
            if time.time() > spawn_start + 3:
                raise Exception('Can\'t spawn a car')

        self.actor_list.append(self.vehicle)

        self.rgb_cam = self.blueprint_library.find('sensor.camera.rgb')
        self.rgb_cam.set_attribute("image_size_x", f"{self.im_width}")
        self.rgb_cam.set_attribute("image_size_y", f"{self.im_height}")
        self.rgb_cam.set_attribute("fov", f"110")

        transform = carla.Transform(carla.Location(x=2.5, z=0.7))
        self.sensor = self.world.spawn_actor(self.rgb_cam, transform, attach_to=self.vehicle)
        self.actor_list.append(self.sensor)
        self.sensor.listen(lambda data: self.process_img(data))

        self.lidar = self.blueprint_library.find('sensor.lidar.ray_cast')
        self.lidar.set_attribute('points_per_second','10000')

        self.lidar_sensor = self.world.spawn_actor(self.lidar, transform, attach_to=self.vehicle)
        self.actor_list.append(self.lidar_sensor)
        self.lidar_sensor.listen(lambda data: self.process_lidar(data))

        self.radar = self.blueprint_library.find('sensor.other.radar')

        self.radar_sensor = self.world.spawn_actor(self.radar, transform, attach_to=self.vehicle)
        self.actor_list.append(self.radar_sensor)
        self.radar_sensor.listen(lambda data: self.process_radar(data))


        # Get the blueprint for the camera
        self.preview_cam = self.world.get_blueprint_library().find('sensor.camera.rgb')
        # Set sensor resolution and field of view
        self.preview_cam.set_attribute('image_size_x', f"{self.im_width}")
        self.preview_cam.set_attribute('image_size_y', f"{self.im_height}")
        self.preview_cam.set_attribute('fov', '110')

        # Set camera sensor relative to a car
        transform = carla.Transform(carla.Location(x=-5, y=0, z=3))

        # Attach camera sensor to a car, so it will keep relative difference to it
        self.preview_sensor = self.world.spawn_actor(self.preview_cam, transform, attach_to=self.vehicle)

        # Register a callback called every time sensor sends a new data
        self.preview_sensor.listen(self._process_preview_img)

        # Add camera sensor to the list of actors
        self.actor_list.append(self.preview_sensor)
        
        self.world.tick()
        self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))
        time.sleep(4)
        self.world.tick()
        colsensor = self.blueprint_library.find("sensor.other.collision")
        self.colsensor = self.world.spawn_actor(colsensor, transform, attach_to=self.vehicle)
        self.actor_list.append(self.colsensor)
        self.colsensor.listen(lambda event: self.collision_data(event))

        #while self.front_camera is None or self.lidar_data is None or self.radar_data is None:
         #   time.sleep(0.01)


        self.episode_start = time.time()
        self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))
        self.world.tick()
        
        while self.img_queue.empty() or self.lidar_queue.empty() or self.radar_queue.empty():
            time.sleep(0.2)
            #print(len(self.actor_list))
        return [self.img_queue.get(), self.lidar_queue.get(), self.radar_queue.get()]

    def collision_data(self, event):
        # What we collided with and what was the impulse
        collision_actor_id = event.other_actor.type_id
        collision_impulse = math.sqrt(event.normal_impulse.x ** 2 + event.normal_impulse.y ** 2 + event.normal_impulse.z ** 2)

        # Filter collisions
        for actor_id, impulse in [['static.sidewalk',-1],['static.road',-1],['vehicle',500]]:
            if actor_id in collision_actor_id and (impulse == -1 or collision_impulse <= impulse):
                return

        # Add collision
        self.collision_hist.append(event)
        

    def process_img(self, image):
        i = np.array(image.raw_data)
        #print(i.shape)
        i2 = i.reshape((self.im_height, self.im_width, 4))
        i3 = i2[:, :, :3]
        #if self.SHOW_CAM:
        #    cv2.imshow("", i3)
         #   cv2.waitKey(1)
        
        self.front_camera = i3
        i3 = np.array(i3.reshape(-1))
        i3 = np.pad(i3, 63488, 'minimum')
        i3 = np.expand_dims(i3, axis=0)
        i3 = np.expand_dims(i3, axis=0)
        self.img_queue.put(i3.reshape(1,-1,1))
        #print("img added")

    def process_lidar(self, data):
        temp = np.frombuffer(data.raw_data, dtype=np.dtype('f4'))
        temp = temp.reshape(-1,1)
        #print(temp.shape)
        points = np.zeros([3000,1], dtype=np.dtype('f4'))
        points[:temp.shape[0]] = temp[:3000]
        points.reshape(-1)
        points = points[:3000:2] / 255
        self.lidar_data = points
        self.lidar_queue.put(points.reshape(1,-1))
        #print("lidar added")

    def process_radar(self, data):
        temp = np.frombuffer(data.raw_data, dtype=np.dtype('f4'))
        temp = temp.reshape(-1,1)
        #print(temp.shape)
        points = np.zeros([500,1], dtype=np.dtype('f4'))
        points[:temp.shape[0]] = temp[:500]
        points.reshape(-1)
        points = points[:500]
        self.radar_data = points
        self.radar_queue.put(points.reshape(1,-1))
        #print("radar added")


    def _process_preview_img(self, image):

        # If camera is disabled - do not process images
        #if self.MAKE_VID is False:
        #    return

        # Get image, reshape and drop alpha channel
        image = np.array(image.raw_data)
        try:
            image = image.reshape((self.im_height, self.im_width, 4))
        except:
            return
        image = image[:, :, :3]

        self.video_writer.write(image)

        if self.SHOW_CAM:
            cv2.imshow("", image)
            cv2.waitKey(1)
        # Set as a current frame in environment
        #self.preview_camera = image
        
    def step(self, action):
        #print(str(action)+',' , end='')
        
        if self.actions[action] != ACTIONS.no_action:
            self.vehicle.apply_control(carla.VehicleControl(throttle=ACTION_CONTROL[self.actions[action]][0], steer=ACTION_CONTROL[self.actions[action]][2]*self.STEER_AMT, brake=ACTION_CONTROL[self.actions[action]][1]))
            
        v = self.vehicle.get_velocity()
        kmh = int(3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2))

        done = False
        
        if len(self.collision_hist) != 0:
            done = True
            reward = -2.5

        # Reward
        elif settings.WEIGHT_REWARDS_WITH_SPEED == 'discrete':
            reward = settings.SPEED_MIN_REWARD if kmh < 50 else settings.SPEED_MAX_REWARD

        elif settings.WEIGHT_REWARDS_WITH_SPEED == 'linear':
            reward = kmh * (settings.SPEED_MAX_REWARD - settings.SPEED_MIN_REWARD) / 100 + settings.SPEED_MIN_REWARD

        elif settings.WEIGHT_REWARDS_WITH_SPEED == 'quadratic':
            reward = (kmh / 50) ** 1.3 * (settings.SPEED_MAX_REWARD - settings.SPEED_MIN_REWARD) + settings.SPEED_MIN_REWARD

        if settings.WEIGHT_REWARDS_WITH_EPISODE_PROGRESS and not done:
            reward *= (time.time() - self.episode_start) / SECONDS_PER_EPISODE

        if self.episode_start + SECONDS_PER_EPISODE < time.time():
            done = True

        if reward > 1:
            reward += 0.25
        if action == 0 or action == 3 or action == 4:
            reward += 0.25
        if action == 5 or action == 6 or action == 7:
            reward -= 0.1

        print(str(action) + " " + str(kmh) + " " + str(reward))

        while self.img_queue.empty() or self.lidar_queue.empty() or self.radar_queue.empty():
            time.sleep(0.2)
        return [self.img_queue.get(), self.lidar_queue.get(), self.radar_queue.get()], reward, done, None


class Agent:
    def __init__(self):
        self.model = self.create_model()
        #self.target_model = self.create_model()
        #self.target_model.set_weights(self.model.get_weights())

        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)

        self.tensorboard = ModifiedTensorBoard(log_dir=f"logs/{MODEL_NAME}-{int(time.time())}")
        #self.target_update_counter = 0
        self.graph = tf.compat.v1.get_default_graph()

        self.terminate = False
        self.last_logged_episode = 0
        self.training_initialized = False
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    def create_model(self):
        #(1,921600)
        input_1 = Input(shape=(1048576,1, ))
        input_2 = Input(shape=(1500, ))
        input_3 = Input(shape=(500, ))

        x = Conv1D(filters=64, kernel_size=5,input_shape=(1048576,1, ), padding='same', kernel_initializer='random_normal', bias_initializer='zeros')(input_1)
        x = Activation('relu')(x)
        x = AveragePooling1D(pool_size=8, strides=8, padding='same')(x)

        x = Conv1D(filters=128, kernel_size=4, padding='same', kernel_initializer='random_normal', bias_initializer='zeros')(x)
        x = Activation('relu')(x)
        x = AveragePooling1D(pool_size=8, strides=8, padding='same')(x)

        x = Conv1D(filters=128, kernel_size=4, padding='same', kernel_initializer='random_normal', bias_initializer='zeros')(x)
        x = Activation('relu')(x)
        x = AveragePooling1D(pool_size=8, strides=8, padding='same')(x)

        x = Conv1D(filters=64, kernel_size=4, padding='same', kernel_initializer='random_normal', bias_initializer='zeros')(x)
        x = Activation('relu')(x)
        x = AveragePooling1D(pool_size=8, strides=8, padding='same')(x)

        x = Flatten()(x)
        
        x = Dense(256, activation='relu', kernel_initializer='random_normal', bias_initializer='zeros')(x)
        x = Dense(128, activation='relu', kernel_initializer='random_normal', bias_initializer='zeros')(x)
        x = Dense(64, activation='relu', kernel_initializer='random_normal', bias_initializer='zeros')(x)

        #x = Flatten()(x)
        #print(x.shape)

        """
        x = Conv2D(64, (5,5), input_shape = (480,270,), padding='same')(input_1)
        x = Activation('relu')(x)
        x = AveragePooling2D(pool_size=(5,5), strides=(3,3), padding='same')(x)

        x = Conv2D(64, (5,5), padding='same')(x)
        x = Activation('relu')(x)
        x = AveragePooling2D(pool_size=(5,5), strides=(3,3), padding='same')(x)

        x = Conv2D(128, (5,5), padding='same')(x)
        x = Activation('relu')(x)
        x = AveragePooling2D(pool_size=(3,3), strides=(2,2), padding='same')(x)

        x = Conv2D(256, (3,3), padding='same')(x)
        x = Activation('relu')(x)
        x = AveragePooling2D(pool_size=(3,3), strides=(2,2), padding='same')(x)

        x = Flatten()(x)
        x = Reshape((64,-1), input_shape=(1,))(x)
        #print(x.shape)
        """



        y = Dense(1024, input_dim = (1500,), activation='relu', kernel_initializer='random_normal', bias_initializer='zeros')(input_2)
        y = Dense(256, activation='relu', kernel_initializer='random_normal', bias_initializer='zeros')(y)
        y = Dense(64, activation='relu', kernel_initializer='random_normal', bias_initializer='zeros')(y)
        y = Dense(32, activation='relu', kernel_initializer='random_normal', bias_initializer='zeros')(y)
        y = Dense(64, activation='relu', kernel_initializer='random_normal', bias_initializer='zeros')(y)
        #y = Reshape((64,-1))(y)
        #print(y.shape)



        z = Dense(256, input_dim = (500,), activation='relu', kernel_initializer='random_normal', bias_initializer='zeros')(input_3)
        z = Dense(128, activation='relu', kernel_initializer='random_normal', bias_initializer='zeros')(z)
        z = Dense(64, activation='relu', kernel_initializer='random_normal', bias_initializer='zeros')(z)
        z = Dense(64, activation='relu', kernel_initializer='random_normal', bias_initializer='zeros')(z)
        #z = Reshape((64,1))(z)
        #print(z.shape)


        merged = concatenate([x,y,z])

        hidden1 = Dense(64, activation='relu', kernel_initializer='random_normal', bias_initializer='zeros')(merged)
	
        hidden2 = Dense(32, activation='relu', kernel_initializer='random_normal', bias_initializer='zeros')(hidden1)

        logits = Dense(NUM_ACTIONS, activation='linear', kernel_initializer='random_normal', bias_initializer='zeros')(hidden2)
        state_values = Dense(1, activation='linear', kernel_initializer='random_normal', bias_initializer='zeros')(hidden2)

        model = Model(inputs=[input_1,input_2,input_3], outputs=[logits, state_values])
        #model.compile(loss="mse", optimizer=Adam(lr=0.001), metrics=["accuracy"])
        return model



        
        #base_model = Xception(weights=None, include_top=False, input_shape=(IM_HEIGHT, IM_WIDTH,3))

        #x = base_model.output
        #x = GlobalAveragePooling2D()(x)

        #predictions = Dense(3, activation="linear")(x)
        #model = Model(inputs=base_model.input, outputs=predictions)
        #model.compile(loss="mse", optimizer=Adam(lr=0.001), metrics=["accuracy"])
        #return model

    def update_replay_memory(self, transition):
        # transition = (current_state, action, reward, new_state, done)
        self.replay_memory.append(transition)

    def train(self):
        #print(len(self.replay_memory))
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return
        # transition = (current_state, action, reward, new_state, done)
        print("training")
        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)
        #minibatch = self.replay_memory

        for batch in minibatch:
            current_state = batch[0]
            action = batch[1]
            reward = batch[2]
            next_state = batch[3]
            done = batch[4]

            with tf.GradientTape() as tape:

                action_probs, critic_value = self.model(current_state)
                action_probs_n, critic_value_n = self.model(next_state)


                diff = reward - critic_value
                action_probs = tf.math.log(action_probs)
                actor_loss = -action_probs*diff
                h = tf.keras.losses.Huber()
                critic_loss = h(tf.expand_dims(critic_value,0),tf.expand_dims(reward,0))
                #critic_loss = tf.keras.losses.Huber(critic_value,reward)

                loss_value = actor_loss + critic_loss
            grads = tape.gradient(loss_value, self.model.trainable_weights)
            self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
            


        log_this_step = False
        if self.tensorboard.step > self.last_logged_episode:
            log_this_step = True
            self.last_log_episode = self.tensorboard.step   
            
            
        

        

        

    def get_qs(self, state):
        #print(state[0].shape)
        #print(state[1].shape)
        #print(state[2].shape)
        return self.model.predict(state)

    def train_in_loop(self):
        X = np.random.uniform(size=(1, IM_HEIGHT, IM_WIDTH, 3)).astype(np.float32)
        y = np.random.uniform(size=(1, 3)).astype(np.float32)
        with self.graph.as_default():
            self.model.fit(X,y, verbose=False, batch_size=1)

        self.training_initialized = True

        while True:
            if self.terminate:
                return
            self.train()
            time.sleep(0.01)

    def save_model(self):
        self.model.save_weights('./checkpoints/saved_weights.h5')

    def load_model(self):
        self.model.load_weights('./checkpoints/saved_weights.h5')



if __name__ == '__main__':
    
    FPS = 5
    # For stats
    ep_rewards = [-200]

    predictions = [0,0,0,0,0,0,0,0,0]
    total_pred = 0

    # For more repetitive results
    #random.seed(1)
    #np.random.seed(1)
    #tf.compat.v1.set_random_seed(1)

    # Memory fraction, used mostly when trai8ning multiple agents
    #gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=MEMORY_FRACTION)
    #backend.set_session(tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options)))

    # Create models folder


    # Create agent and environment
    agent = Agent()
    env = CarEnv()

    #agent.load_model()

    wsettings = env.world.get_settings()
    wsettings.fixed_delta_seconds = 0.1
    wsettings.synchronous_mode = True
    env.world.apply_settings(wsettings)


    # Start training thread and wait for training to be initialized
    #trainer_thread = Thread(target=agent.train_in_loop, daemon=True)
    #trainer_thread.start()
    #while not agent.training_initialized:
    #    time.sleep(0.01)

    # Initialize predictions - forst prediction takes longer as of initialization that has to be done
    # It's better to do a first prediction then before we start iterating over episode steps
    #agent.get_qs(np.ones((env.im_height, env.im_width, 3)))

    # Iterate over episodes
    clear()
    for episode in tqdm(range(1, EPISODES + 1), ascii=True, unit='episodes'):
        #try:
            #time.sleep(5)
            env.collision_hist = []
            agent.replay_memory.clear()

            #if episode%5 == 0:
            #    env.MAKE_VID = True

            # Update tensorboard step every episode
            agent.tensorboard.step = episode

            # Restarting episode - reset episode reward and step number
            episode_reward = 0
            step = 1

            # Reset environment and get initial state
            current_state = env.reset()

            # Reset flag and start iterating until episode ends
            done = False
            env.episode_start = time.time()

            # Play for given number of seconds only
            while True:
                env.world.tick()
                # This part stays mostly the same, the change is to query a model for Q values
                if np.random.random() > epsilon:
                    # Get action from Q table
                    out = agent.get_qs(current_state)
                    action = np.argmax(out[0])
                    print(out[0])
                    print("# ", end=' ')
                    predictions[action] += 1
                    total_pred += 1
                    #if action >= NUM_ACTIONS:
                    #    action = NUM_ACTIONS - 1
                else:
                    # Get random action
                    action = np.random.randint(0, 9)
                    # This takes no time, so we add a delay matching 60 FPS (prediction above takes longer)
                    time.sleep(1/FPS)
                    print("  ", end=' ')

                #print(str(action), end=',')

                new_state, reward, done, _ = env.step(action)

                # Transform new continous state to new discrete state and count reward
                episode_reward += reward

                # Every step we update replay memory
                agent.update_replay_memory((current_state, action, reward, new_state, done))

                current_state = new_state
                step += 1

                if done:
                    break

                

            # End of episode - destroy agents
            #print("End of episode")
            clear()
            agent.train()
            for actor in env.actor_list:
                actor.destroy()

            # Append episode reward to a list and log stats (every given number of episodes)
            ep_rewards.append(episode_reward)
            if not episode % AGGREGATE_STATS_EVERY or episode == 1:
                average_reward = sum(ep_rewards[-AGGREGATE_STATS_EVERY:])/len(ep_rewards[-AGGREGATE_STATS_EVERY:])
                min_reward = min(ep_rewards[-AGGREGATE_STATS_EVERY:])
                max_reward = max(ep_rewards[-AGGREGATE_STATS_EVERY:])
                agent.tensorboard.update_stats(reward_avg=average_reward, reward_min=min_reward, reward_max=max_reward, epsilon=epsilon)

                # Save model, but only when min reward is greater or equal a set value
                if episode%15 == 0 or episode == EPISODES:
                    agent.save_model()

            # Decay epsilon
            if epsilon > MIN_EPSILON:
                epsilon *= EPSILON_DECAY
                epsilon = max(MIN_EPSILON, epsilon)

            #env.MAKE_VID = False

            


    # Set termination flag for training thread and wait for it to finish
    #agent.terminate = True
    #trainer_thread.join()
    print("Predictions")
    if total_pred == 0:
        total_pred += 1
    for i in range(9):
        print(str(i) + ": " + str(predictions[i]*100/total_pred))
    print()
    print(epsilon)
    agent.save_model()



