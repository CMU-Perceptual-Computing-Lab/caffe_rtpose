
# COCO
[[ ! -f coco/pose_iter_440000.caffemodel ]] && wget http://posefs1.perception.cs.cmu.edu/Users/tsimon/Projects/coco/data/models/coco/pose_iter_440000.caffemodel -P coco/  

# MPI
[[ ! -f mpi/pose_iter_264000.caffemodel ]] && wget http://posefs1.perception.cs.cmu.edu/Users/tsimon/Projects/coco/data/models/mpi/pose_iter_160000.caffemodel -P mpi/   

# MPI single level
#[[ ! -f mpi_1l/pose_iter_164000.caffemodel ]] && scp ${SERVER}:/media/posenas1b/Users/zhe/arch/MPI_exp_caffe/pose43/exp08/model/pose_iter_164000.caffemodel mpi_1l/
#[[ ! -f mpi_1l/pose_deploy.prototxt ]] && scp ${SERVER}:/media/posenas1b/Users/zhe/arch/MPI_exp_caffe/pose43/exp08/pose_deploy.prototxt mpi_1l/

