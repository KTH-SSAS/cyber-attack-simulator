sudo echo "hello"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate sim
./run.py
retVal=$?
if [ $retVal -eq 0 ]; then
     sudo poweroff
fi


