







BIN="./build/examples/coco_fast_test/coco_fast_test.bin"

##### things to be set ######
NUM_GPU=3
TESTSET="test2015"
ITERATION=316000
EXPNAME="pose_exp04_zhe_exp14"
SCALE_SEARCH="0.5,1,1.5,2"
#############################

IMAGE_INFO="/home/shihenw/Research/data/COCO/annotations/image_info_${TESTSET}.txt" # note 1k should be removed for entire set
IMAGE_FOLDER="/home/shihenw/Research/data/COCO/${TESTSET}/"

CAFFEMODEL="/home/shihenw/Research/coco_challenge/CocoChallengeTrainTest/caffe_model/${EXPNAME}/pose_iter_${ITERATION}.caffemodel"
PROTOTXT="/home/shihenw/Research/coco_challenge/CocoChallengeTrainTest/pose_exp_caffe/${EXPNAME}/pose_deploy.prototxt"
WRITEFOLDER="/data1/${TESTSET}/${EXPNAME}/pose_iter_${ITERATION}/"


$BIN $NUM_GPU $IMAGE_INFO $IMAGE_FOLDER $CAFFEMODEL $PROTOTXT $WRITEFOLDER $TESTSET $EXPNAME $ITERATION $SCALE_SEARCH
# 0     1           2        3                4         5         6        7          8          9          10

#./build/examples/coco_fast_test/coco_fast_test.bin 3 /home/shihenw/Research/data/COCO/annotations/image_info_val2014_1k.txt /home/shihenw/Research/data/COCO/val2014/

