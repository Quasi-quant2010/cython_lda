# !/bin/sh
python setup.py build_ext --inplace

#OMP_NUM_THREADS=4 \
#python pointer.py \
#-c 0:10 -k 10 -s 2 --alpha=0.5 --beta=0.5 -i 100 > time.txt

for document in  10
do
for iteration in 1
do
tmp1="docoment_$document"
tmp2="_iteration_$iteration"
tmp3=".txt"
OMP_NUM_THREADS=4 python pointer.py -c 0:$document -k 10 -s 2 --alpha=0.5 --beta=0.5 -i $iteration > \
$tmp1$tmp2$tmp3
touch $tmp1$tmp2$tmp3
done
done
