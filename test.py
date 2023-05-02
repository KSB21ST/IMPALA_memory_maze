import gym
import memory_maze
import numpy as np
import skvideo.io
import matplotlib.pyplot as plt

# viridis color list
colors = [
    [253, 231, 37],
    [241, 229, 29],
    [229, 228, 25],
    [216, 226, 25],
    [202, 225, 31],
    [186, 222, 40],
    [173, 220, 48],
    [160, 218, 57],
    [147, 215, 65],
    [132, 212, 75],
    [119, 209, 83],
    [108, 205, 90],
    [96, 202, 96],
    [86, 198, 103],
    [74, 193, 109],
    [64, 189, 114],
    [56, 185, 119],
    [49, 181, 123],
    [41, 175, 127],
    [37, 171, 130],
    [33, 166, 133],
    [31, 161, 135],
    [30, 157, 137],
    [31, 151, 139],
    [32, 146, 140],
    [33, 142, 141],
    [35, 137, 142],
    [37, 131, 142],
    [39, 127, 142],
    [41, 122, 142],
    [43, 117, 142],
    [45, 113, 142],
    [47, 107, 142],
    [49, 102, 142],
    [52, 97, 141],
    [54, 92, 141],
    [57, 85, 140],
    [60, 80, 139],
    [62, 74, 137],
    [64, 69, 136],
    [66, 63, 133],
    [69, 56, 130],
    [70, 50, 126],
    [71, 44, 122],
    [72, 37, 118],
    [72, 29, 111],
    [72, 23, 105],
    [71, 16, 99],
    [70, 8, 92],
    [68, 1, 84],
]  

def save_video(video, dir):
    fps = '4'
    crf = '17'
    vid_out = skvideo.io.FFmpegWriter(f'{dir}.mp4', 
                inputdict={'-r': fps},
                outputdict={'-r': fps, '-c:v': 'libx264', '-crf': crf, 
                            '-preset': 'ultrafast', '-pix_fmt': 'yuv444p'}
    )

    for frame in video:
        vid_out.writeFrame(frame)
    vid_out.close()
    

def generate_video(image, maze_layout, agent_pos, agent_dir, top_camera):
    
    l, bx, by = maze_layout.shape
    _, h, w, c = image.shape
    assert h==w and (h==256 or h==64)
    assert bx==by
    agent_pose = np.concatenate((agent_pos, agent_dir), axis=1)  # [l,4]
    margin = 32 if h == 256 else 8
    h_fig = h - margin * 2  # actual top camera size

    # draw pose on topdown
    dir_len = 20 if h == 256 else 10
    for j in range(l):
        
        # draw current pose
        for i in range(j+1):
            x = round(agent_pos[i, 0] * h_fig / bx) + margin
            y = h - round(agent_pos[i, 1] * h_fig / by) - margin
            
            if i == j:  # current pose with direction
                for n in range(dir_len):
                    x_ = x + round(agent_dir[i, 0] * n)
                    y_ = y - round(agent_dir[i, 1] * n)
                    if x_ < h and x_ > 0 and y_ < h and y_ > 0:
                        top_camera[j, y_, x_] = [0, 0, 0]
                            
            top_camera[j, y, x] = [255, 0, 0]

    # video = np.concatenate((topdown, scenes), axis=2)
    video = np.concatenate((top_camera, image), axis=2)
    return video


# Set this if you are getting "Unable to load EGL library" error:
#  os.environ['MUJOCO_GL'] = 'glfw'  

env = gym.make(f'MemoryMaze-Custom-v0')
# env = gym.make('memory_maze:MemoryMaze-9x9-v0')

ep = {'image': [], 'agent_pos': [], 'agent_dir': [], 'top_camera': [], 'maze_layout': []}

o = env.reset()
for k, v in ep.items():
    v.append(o[k])

while True:
    o, r, d, info = env.step(1)
    for k, v in ep.items():
        v.append(o[k])
    if d: 
        break

for k, v in ep.items():
    ep[k] = np.array(v)

video = generate_video(image=ep['image'], maze_layout=ep['maze_layout'], agent_pos=ep['agent_pos'], agent_dir=ep['agent_dir'], top_camera=ep['top_camera'])
plt.imsave('./traj0.png', video[0])
save_video(video, dir='./traj')