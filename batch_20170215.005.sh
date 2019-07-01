#!/bin/bash

pids=(0 0 0 0 0 0 0 0 0 0)

#short experiments first
python -u derive_from_fitted.py configs/config_20170302.002_lp_1min-cal_0.ini 20170302.002_lp_1min-cal.h5 >& logs/log_20170302.002_lp_1min-cal_0.out &
pids[0]=$!
sleep 5
python -u derive_from_fitted.py configs/config_20170302.002_lp_1min-cal_1.ini 20170302.002_lp_1min-cal.h5 >& logs/log_20170302.002_lp_1min-cal_1.out &
pids[1]=$!
sleep 5
python -u derive_from_fitted.py configs/config_20170302.002_lp_1min-cal_2.ini 20170302.002_lp_1min-cal.h5 >& logs/log_20170302.002_lp_1min-cal_2.out &
pids[2]=$!
sleep 5
python -u derive_from_fitted.py configs/config_20170302.002_lp_1min-cal_3.ini 20170302.002_lp_1min-cal.h5 >& logs/log_20170302.002_lp_1min-cal_3.out &
pids[3]=$!
sleep 5
python -u derive_from_fitted.py configs/config_20170302.002_lp_1min-cal_4.ini 20170302.002_lp_1min-cal.h5 >& logs/log_20170302.002_lp_1min-cal_4.out &
pids[4]=$!
sleep 5
python -u derive_from_fitted.py configs/config_20170302.002_lp_1min-cal_5.ini 20170302.002_lp_1min-cal.h5 >& logs/log_20170302.002_lp_1min-cal_5.out &
pids[5]=$!
sleep 5
python -u derive_from_fitted.py configs/config_20170302.002_lp_1min-cal_6.ini 20170302.002_lp_1min-cal.h5 >& logs/log_20170302.002_lp_1min-cal_6.out &
pids[6]=$!
sleep 5
python -u derive_from_fitted.py configs/config_20170302.002_lp_1min-cal_7.ini 20170302.002_lp_1min-cal.h5 >& logs/log_20170302.002_lp_1min-cal_7.out &
pids[7]=$!
sleep 5
python -u derive_from_fitted.py configs/config_20170302.002_lp_1min-cal_8.ini 20170302.002_lp_1min-cal.h5 >& logs/log_20170302.002_lp_1min-cal_8.out &
pids[8]=$!
sleep 5
python -u derive_from_fitted.py configs/config_20170302.002_lp_1min-cal_9.ini 20170302.002_lp_1min-cal.h5 >& logs/log_20170302.002_lp_1min-cal_9.out &
pids[9]=$!
sleep 5
wait ${pids[0]} ${pids[1]} ${pids[2]} ${pids[3]} ${pids[4]} ${pids[5]} ${pids[6]} ${pids[7]} ${pids[8]} ${pids[9]} 
