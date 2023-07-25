CHECKPOINT_PATH=/home/SVSstudents/Downloads/Project/gym-pybullet-drones/assignment1/results
cd /home/SVSstudents/Downloads/Project/gym-pybullet-drones
pip install -e .
cd /home/SVSstudents/Downloads/Project/gym-pybullet-drones/assignment1
vglrun -d /dev/dri/card1 python ReachThePoint.py --exp $1 --workers 1 --gui 1