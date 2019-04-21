#!/bin/bash

for i in {1..4}
do
rl -c 100000 partisan-dem-[$i].txt >> dem_sample.txt
done
end

for i in {1..11}
do
rl -c 50000 partisan-rep-[$i].txt >> rep_sample.txt
done
end