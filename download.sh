DATA_DIR=./dataset/movielens
SIZE=100k
mkdir -p ${DATA_DIR}-${SIZE}
wget http://files.grouplens.org/datasets/movielens/ml-${SIZE}.zip -O ${DATA_DIR}-${SIZE}/ml-${SIZE}.zip
unzip ${DATA_DIR}-${SIZE}/ml-${SIZE}.zip -d ${DATA_DIR}-${SIZE}
mv ${DATA_DIR}-${SIZE}/ml-${SIZE}/* ${DATA_DIR}-${SIZE}/
rm -r ${DATA_DIR}-${SIZE}/ml-${SIZE}
rm ${DATA_DIR}-${SIZE}/ml-${SIZE}.zip
