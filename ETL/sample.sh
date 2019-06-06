#!/bin/bash

for i in {1..4}
do
rl -c 300000 ./data/partisan_data/partisan-dem-[$i].txt >> ./data/partisan_data/dem_sample_0518.txt
done
end

for i in {1..11}
do
rl -c 150000 ./data/partisan_data/partisan-rep-[$i].txt >> ./data/partisan_data/rep_sample_0518.txt
done
end 