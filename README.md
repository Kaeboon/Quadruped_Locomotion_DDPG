# Quadruped Robot Locomotion with Reinforcement Learning

This is our final project for ME5418: Machine Learning in Robotics. \
In this project, we apply Deep Deterministic Policy Gradient (DDPG) to teach a Aliengo quadruped robot to move inside a PyBullet Simulation environment.

## Dependencies

- Python 3.6.10
- pip:
    - tensorflow==1.11.0
    - gym==0.21.0
    - pyglet==1.5.27
    - imageio
    - matplotlib
    - scipy
    - numpy
    - pybullet==3.2.5
    - jupyter

The above dependencies has been specified in the `requirements.yml` file.

To install them inside a conda environment, follow the steps:
- `cd` into your script directory
- Run the following command

```
conda env create -n ME5418_Group18 -f requirements.yml
conda activate ME5418_Group18
```

## Trained policy

We have trained three different policy.  See the `Trained_Policy_Video.mp4` for the demo.
- Policy 1: A local maximum suboptimal policy. Robot falls forward to the ground.
- Policy 2: Using simplified reward structure. Robot can crawl forward but slowly deviates from path.
- Policy 3: Without gravity. Robot can push itself forward slowly.

## Usage

I have provided two version of code, one in python format - `main_code.py`, and one in jupyter notebook format - `main_code.ipynb`. You can use either one to run the training or testing.

### `main_code.py`:

- In the terminal, `cd` into the code directory.

To test our trained policy:

- Run
```
python main_code.py --TRAINING 0 --Policy 1
```
where you set the `TRAINING` argument to `0`, and set the `Policy` argument to either `1` or `2` or `3` depending on which policy you want to test.

To train from scratch:

- Run
```
python main_code.py  --TRAINING 1 --Policy 0
```
where you set the `TRAINING` argument to `1`, and set the `Policy` argument to `0`.

- In a seperate terminal, `cd` into your script directory and run the following command to view the agent performance from Tensorboard:
```
tensorboard --logdir tf_summary_save
```

### `main_code.ipynb`:

- In the terminal, `cd` into the code directory.
- Open the `main_code.ipynb` file by running the following command:
```
jupyter notebook main_code.ipynb
```

To test our trained policy:

- Inside the jupyter notebook's first code cell, set `TRAINING` variable to `False`, and set the `Policy` variable to `1` or `2` or `3`. For example:
    ```
    TRAINING = False
    Policy = 2
    ```
- Then, run through every cell until the `Main code Cell` to start the testing process.

To train from scratch:
- Inside the jupyter notebook's first code cell, set `TRAINING` variable to `True`, and set the `Policy` variable to `0`. For example:
    ```
    TRAINING = True
    Policy = 0
    ```
- Then, run through every cell until the `Main code Cell` to start the training process.
- In a seperate terminal, `cd` into your script directory and run the following command to view the agent performance from Tensorboard:
```
tensorboard --logdir tf_summary_save
```
- To stop the training, simply stop the `Main code Cell`, and run the `Disconnect Cell` to stop the Pybullet Simulation.

## Key Files and Folders
- `aliengo_gym.py` - Define the environment class
- `DDPGNet.py` - Define the Neural network class
- `main_code.ipynb` - Define parameters, Driver of program
- `main_code.py` - Python version of `main_code.ipynb`
- `pybullet_data/` - Folder that stores 3D model information of Aliengo and Plane
- `gifs_save/` - Folder that stores the GIFs of the render
- `model_save/` - Folder that stores model parameter
- `tf_summary_save/` - Folder that stores tf summary for tensorboard visualisation
- `episode_buffer_save/` - Folder that stores the replay buffer information
- `README.so` - A readme file with instructions on how to run our code
- `Trained_Policy_Video.mp4` - Video showcasing our trained policy
- `report/` - Folder that stores my final report for ME5418

## Author
- HO KAE BOON (E0550469@u.nus.edu)
- ZENG ZHENGTAO (E1192691@u.nus.edu)
- HUANG WENBO (E1192643@u.nus.edu)
- ZHONG HAO (E1192473@u.nus.edu)
