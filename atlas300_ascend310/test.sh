output="output.txt"
#rm $output
#for dir in $(ls convert_data/)
#for line in $(<test_1shot.txt)
cat test_5shot.txt | while read line
do
    python3 detect.py $line
    rm $output
done
