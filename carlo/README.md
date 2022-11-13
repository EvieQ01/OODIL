## Running OOD-IL for Driving
Instructions are in `setup/` directory.
```bash
cd setup
bash run.sh 
```
You can change the parameter setting in `run.sh`

## Demonstrations for Driving
Due to the space limit, we included the demonstrations [here](https://drive.google.com/drive/folders/1zR-6VwiA7ev8PYuTo8hK3Z_-mNqZyRzQ?usp=sharing). Download `demo_driving.zip`. And unzip it here.

Specifically, we make expert demonstrations with manually designed policy, and you can also make your own demonstrations by:
```bash
cd CARLO_x
python example_obstacle.py
```
You can change the environment setting in `example_obstacle.py`
