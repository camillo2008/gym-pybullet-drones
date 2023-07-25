#!/bin/bash

rm -rf easy
rm -rf medium
rm -rf hard

mkdir easy
mkdir medium
mkdir hard
declare -a arr=("easy" "medium" "hard")
for ((i=1; i<=3; i++))
do
  DIFF=0.5
  RES=$(echo "scale=4; $DIFF*$i" | bc)
  ./generate_multiple.bash 100 "$RES"
  mv ./generated_envs/* "${arr[$i-1]}"
done

mv ./easy ./generated_envs
mv ./medium ./generated_envs
mv ./hard ./generated_envs