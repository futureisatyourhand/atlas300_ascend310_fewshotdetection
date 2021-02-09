# ./main  -iter_time  -batch_size  -channel -wid -hig
###############################################################
# protein_cnn app
#./main 100 2 21 1 17  ../data/5000_X_val.npy float64


# D3Predict-part1 app
#time ./main 100 64 42 1 512  ../data/d3_data.npy float32


# D3Predict-part1 app
#time ./main  16 64 2 128 3  ../data/diher.npy float32


# ContactPred app
#time ./main 24 16 109 109 74 ../data/contact_data.npy  float32


# bioavailability_model app
time ./main 1 1 3 416 416 ../data/atoms.npy  float32
