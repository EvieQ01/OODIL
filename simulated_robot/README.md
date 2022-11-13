
## INSTALLATION
Run:
```bash
pip install -r requirements.txt
```
Note that you need a MuJoCo license. Please follow the instruction in [mujoco-py](https://github.com/openai/mujoco-py
) for help.
## Running OOD-IL for Simulated Franka Panda Arm
```bash
bash run_set1.sh
```
## Demonstrations for Franka Panda Arm
Due to the space limit, we included the demonstrations [here](https://drive.google.com/drive/folders/1zR-6VwiA7ev8PYuTo8hK3Z_-mNqZyRzQ?usp=sharing). Download `demo_panda_new.zip`. And unzip it here.
Specifically, we make expert demonstrations with manually designed policy, and you can also make your own demonstrations by:
```bash
python collect_demo_panda.py
```
You can change the environment setting in `pdenv/gym_panda`

