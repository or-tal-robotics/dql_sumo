import numpy as np
class ReplayMemory():
    def __init__(self, size = 50000, frame_height = 84, fram_width = 84, agent_history_lenth = 4, batch_size = 32):
        self.size = size
        self.frame_height = frame_height
        self.frame_width = fram_width
        self.agent_history_lenth = agent_history_lenth
        self.batch_size = batch_size
        self.count = 0
        self.current = 0
        
        self.actions = np.empty(self.size, dtype = np.int32)
        self.rewards = np.empty(self.size, dtype = np.float32)
        self.frames = np.empty((self.size,self.frame_height, self.frame_width), dtype = np.uint8)
        self.terminal_flags = np.empty(self.size, dtype = np.bool)
        self.states = np.empty((self.batch_size,self.agent_history_lenth,self.frame_height,self.frame_width), dtype=np.uint8)
        self.new_states = np.empty((self.batch_size,self.agent_history_lenth,self.frame_height,self.frame_width), dtype=np.uint8)
        self.indices = np.empty(self.batch_size, dtype = np.int32)
        
    def add_experience(self, action, frame, reward, terminal):
        if frame.shape != (self.frame_height, self.frame_width):
            raise ValueError('Frames dimansions are wrong!')
        
        self.actions[self.current] = action
        self.frames[self.current, ...] = frame
        self.rewards[self.current] = reward
        self.terminal_flags[self.current] = terminal
        
        self.count = max(self.count , self.current + 1)
        self.current = (self.current + 1) % self.size
        
    def get_state(self, index):
        if self.count is 0:
            raise ValueError("The replay memory is empty!")
        if index < self.agent_history_lenth-1:
            raise ValueError("Index must be over 3!")
        return self.frames[index-self.agent_history_lenth+1:index+1, ...]
    
    def get_valid_indices(self):
        for i in range(self.batch_size):
            while True:
                index = np.random.randint(self.agent_history_lenth, self.count - 1)
                if index < self.agent_history_lenth:
                    continue
                if index >= self.current and index - self.agent_history_lenth <= self.current:
                    continue
                if self.terminal_flags[index - self.agent_history_lenth].any():
                    continue
                break
            self.indices[i] = index
    
    def get_minibatch(self):
        if self.count < self.agent_history_lenth:
            raise ValueError('Not enough history to get a minibatch!')
        
        self.get_valid_indices()
        for i, idx in enumerate(self.indices):
            self.states[i] = self.get_state(idx - 1)
            self.new_states[i] = self.get_state(idx)
        
        return np.transpose(self.states, axes=(0,2,3,1)), \
            self.actions[self.indices], \
            self.rewards[self.indices], \
            np.transpose(self.new_states, axes=(0,2,3,1)), \
            self.terminal_flags[self.indices]
            

def update_state(state, obs_small):
    return np.append(state[:,:,1:], np.expand_dims(obs_small, 2), axis = 2)

def learn(model, target_model, experience_replay_buffer, gamma, batch_size):
    states, actions, rewards, next_states, dones = experience_replay_buffer.get_minibatch()
    next_Qs = target_model.predict(next_states)
    next_Q = np.amax(next_Qs, axis=1)
    targets = rewards + np.invert(dones).astype(np.float32) * gamma * next_Q
     
    loss = model.update(states, actions, targets)
    return loss


def make_video(images, outimg=None, fps=5, size=None,
               is_color=True, format="XVID"):
    """
    Create a video from a list of images.
 
    @param      outvid      output video
    @param      images      list of images to use in the video
    @param      fps         frame per second
    @param      size        size of each frame
    @param      is_color    color
    @param      format      see http://www.fourcc.org/codecs.php
    @return                 see http://opencv-python-tutroals.readthedocs.org/en/latest/py_tutorials/py_gui/py_video_display/py_video_display.html
 
    The function relies on http://opencv-python-tutroals.readthedocs.org/en/latest/.
    By default, the video will have the size of the first image.
    It will resize every image to this size before adding them to the video.
    """
    from cv2 import VideoWriter, VideoWriter_fourcc, imread, resize
    fourcc = VideoWriter_fourcc(*format)
    vid = None
    for image in images:
        if not os.path.exists(image):
            raise FileNotFoundError(image)
        img = imread(image)
        if vid is None:
            if size is None:
                size = img.shape[1], img.shape[0]
            vid = VideoWriter(outvid, fourcc, float(fps), size, is_color)
        if size[0] != img.shape[1] and size[1] != img.shape[0]:
            img = resize(img, size)
        vid.write(img)
    vid.release()