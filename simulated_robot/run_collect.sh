init=2.0
python  collect_demo_panda.py  --env_id panda-v0 --buffer_size 100000 --render --init_range ${init}
python  collect_demo_panda.py  --env_id disabledpanda-v1 --buffer_size 100000 --render --init_range ${init}
python  collect_demo_panda.py  --env_id disabledpanda-v3 --buffer_size 100000 --render --init_range ${init}
python  collect_demo_panda.py  --env_id disabledpanda-v134 --buffer_size 100000 --render --init_range ${init}
python  collect_demo_panda.py  --env_id disabledpanda-v1346 --buffer_size 100000 --render --init_range ${init}
python  collect_demo_panda.py  --env_id disabledpanda-v4 --buffer_size 100000 --render --init_range ${init}
python  collect_demo_panda.py  --env_id disabledpanda-v6 --buffer_size 100000 --render --init_range ${init}
python  collect_demo_panda.py  --env_id disabledpanda-v14 --buffer_size 100000 --render --init_range ${init}
