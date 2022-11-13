## xml indicates the dynamic
## --dump, save the cluster
## --render, make a video
xml=12.0
do
    python main.py --env-name "CustomHopper-v0" --xml hopper_${xml}.xml 
    python save_traj_trpo.py --env-name "CustomHopper-v0" --xml hopper_${xml}.xml --dump --render
done

xml=19.9
do
    python main.py --env-name "CustomWalker2d-v0" --xml walker2d_${xml}.xml 
    python save_traj_trpo.py --env-name "CustomHopper-v0" --xml hopper_${xml}.xml --dump --render
done

xml=half_cheetah
do
    python main.py --env-name "CustomHopper-v0" --xml half_cheetah.xml 
    python save_traj_trpo.py --env-name "CustomHopper-v0" --xml half_cheetah.xml --dump --render
done
