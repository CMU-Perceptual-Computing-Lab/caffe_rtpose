SERVER=gpuserver2.perception.cs.cmu.edu

# COCO
[[ ! -f coco/pose_iter_440000.caffemodel ]] && scp ${SERVER}:/media/posefs4b/User/zhe/arch/COCO_exp_caffe/pose56/exp22/model/pose_iter_440000.caffemodel coco/

# MPI
[[ ! -f mpi/pose_iter_264000.caffemodel ]] && scp ${SERVER}:/media/posenas1b/Users/zhe/arch/MPI_exp_caffe/pose43/exp04/model/pose_iter_264000.caffemodel mpi

