## INSTALLATION

Codes for `HalfCheetah`, `Hopper`, `Walker2d` environment.

Run:
```bash
pip install -e requirements.txt
```
to install.


Then install the environments:

```bash
cd all_envs
pip install -e .
```

## Running OOD-IL for MuJoCo
Instructions are in `setup1/` directory.
```bash
cd setup1

bash run_cheetah.sh # for half_cheetah environment
bash run_hopper.sh # for hopper environment
bash run_walker.sh # for walker2d environment
```
## Demonstrations for MuJoCo
Due to the space limit, we included the demonstrations [here](https://drive.google.com/drive/folders/1zR-6VwiA7ev8PYuTo8hK3Z_-mNqZyRzQ?usp=sharing). Download `demo_mujoco.zip`. And unzip it here.

Specifically, we train expert with `trpo`, you can also make your own demonstrations by:
```bash
cd pytorch-trpo-master
bash run_make_demo.sh
```
You can change the environment setting in `run_make_demo.sh`
